"""DownstreamEvaluator — the per-task evaluation loop for the prediction track.

``run_eval`` sets up the data provider + segment binder and the model, then hands
them here. For each task this binds the cohort's eligible data, runs the model
(``Encoder`` → the uniform PCA probe, or ``Predictor`` → direct predictions), and
scores the test split.
"""

from __future__ import annotations

import logging

import numpy as np

from downstream_evaluation.config import ClassifierConfig
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


def _metrics_for(task: str, y_true, y_pred) -> dict[str, float]:
    ttype = get_task_type(task)
    if ttype == "binary":
        return compute_binary_metrics(y_true, y_pred)
    # multiclass/ordinal scores discrete class predictions. The uniform ordinal probe
    # already predicts ints; a Predictor may hand back raw floats (e.g. the hybrid's
    # rank-combined scores), so round to int before scoring (a no-op when predictions
    # are already discrete).
    if ttype == "multiclass":
        return compute_multiclass_metrics(y_true, np.round(y_pred).astype(int))
    if ttype == "ordinal":
        return compute_ordinal_metrics(y_true, np.round(y_pred).astype(int))
    return compute_regression_metrics(y_true, y_pred)


def _is_encoder(model) -> bool:
    """An Encoder produces embeddings (``encode`` per participant, or ``encode_cohort``
    for the cohort matrix); a Predictor produces predictions (``fit``/``predict``).
    Encoder takes priority if a model exposes both."""
    return hasattr(model, "encode") or hasattr(model, "encode_cohort")


def _public_inputs(td):
    """Per-participant data as the public ``(n_segments, 24, 38)`` arrays for a unified
    :class:`~openmhc.Method`, or ``None`` when the model self-serves from its own cache
    (``needs_segments=False`` → ``td.inputs`` is ``None``)."""
    if td.inputs is None:
        return None
    return [seg.as_array() for seg in td.inputs]


def _set_context(model, td) -> None:
    """Hand a unified Method its per-(task, split) context, if it defines the hook.

    Mirrors the ``set_temporal_window`` duck-typed hook: external Methods omit it and
    never see cohort identity; bundled baselines read ``user_ids`` / ``task`` from it."""
    if hasattr(model, "set_context"):
        from openmhc._protocols import EvalContext

        model.set_context(
            EvalContext(task=td.task, split=td.split, user_ids=td.user_ids, dates=td.dates)
        )


class DownstreamEvaluator:
    """Run the per-task fit→predict→score loop for one model."""

    def __init__(
        self,
        seed: int = 42,
        pca_n_components: int | None = 50,
        predictions_dir: str | None = None,
    ):
        """Args:
        seed: random_state for the probe / model.
        pca_n_components: PCA dim for the encoder probe (``None`` to disable).
        predictions_dir: when set, write each task's test predictions (uid, y_true,
            y_pred, y_proba) to ``<predictions_dir>/<method>/<task>/test.parquet``.
        """
        self.seed = seed
        self.pca_n_components = pca_n_components
        self.predictions_dir = predictions_dir

    def run(self, provider, binder, model, tasks: list[str]) -> dict[str, dict]:
        """Evaluate ``model`` on each task; returns ``{task: {**metrics, n_test}}``."""
        results: dict[str, dict] = {}
        for task in tasks:
            train_td = provider.task_data(task, "train")
            test_td = provider.task_data(task, "test")
            if binder is not None:
                train_td, test_td = binder.bind(train_td), binder.bind(test_td)
            if len(train_td.user_ids) == 0 or len(test_td.user_ids) == 0:
                logger.warning("task %s: empty train/test split, skipping", task)
                continue
            y_true, y_pred = self._eval_task(model, task, train_td, test_td)
            results[task] = {**_metrics_for(task, y_true, y_pred), "n_test": int(len(y_true))}
            if self.predictions_dir is not None:
                from downstream_evaluation.evaluation.predictions_io import (
                    write_task_predictions,
                )

                write_task_predictions(
                    self.predictions_dir,
                    getattr(model, "name", type(model).__name__),
                    task,
                    test_td.user_ids,
                    y_true,
                    y_pred,
                )
            logger.info("  %s: n_test=%d", task, len(y_true))
        return results

    def _eval_task(self, model, task: str, train_td, test_td):
        """Evaluate one task; returns ``(y_true, y_pred)`` for the test split.

        Encoder branch: embed each participant → fit PCA + the uniform probe on
        train → predict test. Predictor branch: ``fit``/``predict`` end-to-end on
        the bound ``TaskData``.
        """
        ttype = get_task_type(task)
        if getattr(model, "predicts_from_arrays", False):
            # Unified Method: clean per-participant arrays + labels + task_type. The
            # optional set_context hook hands bundled baselines their cohort identity
            # (user_ids / task) before each call; external methods omit it.
            train_data = _public_inputs(train_td)
            test_data = _public_inputs(test_td)
            _set_context(model, train_td)
            model.fit(train_data, train_td.labels, ttype)
            _set_context(model, test_td)
            y_pred = np.asarray(model.predict(test_data))
            return test_td.labels, y_pred
        if _is_encoder(model):
            if hasattr(model, "encode_cohort"):
                # Cache-based encoders return the cohort's per-user embedding matrix
                # directly (build-on-miss/load-on-hit lives inside the model).
                Xtr = np.asarray(model.encode_cohort(task, train_td), dtype=np.float32)
                Xte = np.asarray(model.encode_cohort(task, test_td), dtype=np.float32)
            else:
                Xtr = np.stack(
                    [np.asarray(model.encode(x), dtype=np.float32).ravel() for x in train_td.inputs]
                )
                Xte = np.stack(
                    [np.asarray(model.encode(x), dtype=np.float32).ravel() for x in test_td.inputs]
                )
            for X in (Xtr, Xte):
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            clf_cfg = ClassifierConfig(
                type=_PROBE_BY_TASKTYPE[ttype],
                use_scaler=False,
                pca_n_components=self.pca_n_components,
            )
            clf = create_model(clf_cfg, random_state=self.seed, task_type=ttype)
            clf.fit(Xtr, train_td.labels)
            y_pred = clf.predict_proba(Xte)[:, 1] if ttype == "binary" else clf.predict(Xte)
        else:  # Predictor — fits + predicts end-to-end on the bound TaskData
            model.fit(train_td)
            y_pred = np.asarray(model.predict(test_td))
        return test_td.labels, y_pred
