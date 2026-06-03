"""Prediction engine — provider → model → (uniform probe) → metrics.

Mirrors the imputation/forecasting ``run_eval(config, model)`` shape, and powers
both surfaces: Surface 1 (an external model) and Surface 2 (our bundled baselines)
run through this same engine. The model implements one of the ``openmhc`` protocols:

  - ``Encoder``   — ``encode(data) -> (D,)`` embedding per participant. The engine
                    fits a *uniform* PCA-50 linear probe on the embeddings and scores
                    it, so the result reflects the representation, not the probe.
  - ``Predictor`` — ``fit(data, labels)`` + ``predict(data) -> preds``. End-to-end;
                    the engine scores its predictions directly.

All cohort / temporal / label logic comes from :class:`TaskDataProvider` (the
embedded-temporal lookup). The model only ever sees a participant's *eligible*
data, at the granularity it declares via ``input_granularity`` (default series).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from downstream_evaluation.config import ClassifierConfig
from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
from downstream_evaluation.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_ordinal_metrics,
    compute_regression_metrics,
    get_task_type,
)
from downstream_evaluation.models.registry import create_model

logger = logging.getLogger(__name__)

# Uniform linear probe per task type — the same probe for every encoder, so a
# method's score reflects its representation rather than its choice of classifier.
_PROBE_BY_TASKTYPE: dict[str, str] = {
    "binary": "logistic_regression",
    "multiclass": "logistic_regression",
    "ordinal": "logreg_ordinal",
    "regression": "linear_regression",
}


@dataclass
class EvalConfig:
    """Config for the prediction engine.

    Args:
        data_dir: dataset root (its ``processed/`` holds the lookups + sensor data).
        split_users: ``{"train"/"validation"/"test": [user_id, ...]}``.
        tasks: tasks to evaluate.
        seed: random_state for the probe / model.
        pca_n_components: PCA dim for the encoder probe (``None`` to disable).
    """

    data_dir: str
    split_users: dict
    tasks: list[str] = field(default_factory=list)
    seed: int = 42
    pca_n_components: int | None = 50


def _metrics_for(task: str, y_true, y_pred) -> dict[str, float]:
    ttype = get_task_type(task)
    if ttype == "binary":
        return compute_binary_metrics(y_true, y_pred)
    if ttype == "multiclass":
        return compute_multiclass_metrics(y_true, y_pred)
    if ttype == "ordinal":
        return compute_ordinal_metrics(y_true, y_pred)
    return compute_regression_metrics(y_true, y_pred)


def _is_encoder(model) -> bool:
    """An Encoder produces embeddings (``encode``); a Predictor produces predictions
    (``fit``/``predict``). Encoder takes priority if a model exposes both."""
    return hasattr(model, "encode")


def eval_task(model, task: str, train_td, test_td, seed: int, pca_n: int | None):
    """Evaluate one task; returns ``(y_true, y_pred)`` for the test split.

    Encoder branch: embed each participant → fit PCA-``pca_n`` + the uniform probe on
    train → predict test. Predictor branch: ``fit`` on train then ``predict`` test.
    """
    ttype = get_task_type(task)
    if _is_encoder(model):
        Xtr = np.stack([np.asarray(model.encode(x), dtype=np.float32).ravel() for x in train_td.inputs])
        Xte = np.stack([np.asarray(model.encode(x), dtype=np.float32).ravel() for x in test_td.inputs])
        for X in (Xtr, Xte):
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        clf_cfg = ClassifierConfig(
            type=_PROBE_BY_TASKTYPE[ttype], use_scaler=False, pca_n_components=pca_n
        )
        clf = create_model(clf_cfg, random_state=seed, task_type=ttype)
        clf.fit(Xtr, train_td.labels)
        y_pred = clf.predict_proba(Xte)[:, 1] if ttype == "binary" else clf.predict(Xte)
    else:  # Predictor
        model.fit(train_td.inputs, train_td.labels)
        y_pred = np.asarray(model.predict(test_td.inputs))
    return test_td.labels, y_pred


def run_eval(config: EvalConfig, model) -> dict[str, dict]:
    """Run the prediction eval for one model (``Encoder`` or ``Predictor``).

    Builds a :class:`TaskDataProvider` at the model's declared granularity and, per
    task, hands the model the cohort's eligible data and scores the result.

    Returns ``{task: {**metrics, "n_test": int}}``.
    """
    grain = getattr(model, "input_granularity", "series")
    lookup = f"{config.data_dir}/processed/{LOOKUP_BY_GRANULARITY[grain]}"
    provider = TaskDataProvider(lookup, config.split_users, granularity=grain)
    logger.info("Running prediction eval (granularity=%s) on %d tasks", grain, len(config.tasks))

    results: dict[str, dict] = {}
    for task in config.tasks:
        train_td = provider.task_data(task, "train")
        test_td = provider.task_data(task, "test")
        if len(train_td.user_ids) == 0 or len(test_td.user_ids) == 0:
            logger.warning("task %s: empty train/test split, skipping", task)
            continue
        y_true, y_pred = eval_task(model, task, train_td, test_td, config.seed, config.pca_n_components)
        results[task] = {**_metrics_for(task, y_true, y_pred), "n_test": int(len(y_true))}
        logger.info("  %s: n_test=%d", task, len(y_true))
    return results


def score_predictions(preds: dict) -> dict[str, dict]:
    """Score pre-computed per-task predictions (the baked dev-repro path).

    ``preds`` maps ``task -> TaskPrediction(y_true, y_pred)`` (e.g. from the baked
    ``LinearProbeMethod`` / ``XGBoostMethod``). Returns ``{task: {**metrics, n_test}}``.
    """
    return {
        t: {**_metrics_for(t, tp.y_true, tp.y_pred), "n_test": int(len(tp.y_true))}
        for t, tp in preds.items()
    }
