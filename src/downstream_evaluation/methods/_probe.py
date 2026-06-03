"""Shared per-task fit→predict loop for the probe methods (linear + xgboost).

Both ``LinearProbeMethod`` and ``XGBoostMethod`` load the same baked
``features/<name>/<split>.parquet`` tables and fit one probe per task; they
differ only in the classifier registry keys (linear vs tree) and whether a
scaler is applied. This helper holds that common loop.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.config import ClassifierConfig
from downstream_evaluation.evaluation.metrics import get_task_type
from downstream_evaluation.methods.base import TaskPrediction
from downstream_evaluation.models.registry import create_model

logger = logging.getLogger(__name__)


def fit_predict_tables(
    name: str,
    features_dir: str,
    tasks: list[str],
    classifiers: dict[str, str],
    base_config: ClassifierConfig,
    seed: int = 42,
) -> dict[str, TaskPrediction]:
    """Load the baked train/test tables and fit→predict one probe per task.

    Args:
        name: method/folder under ``features_dir``.
        features_dir: root holding ``<name>/<split>.parquet``.
        tasks: tasks to evaluate.
        classifiers: task-type → classifier registry key.
        base_config: ClassifierConfig template carrying the eval's exact recipe
            (scaler on/off, the xgboost hyperparameters, etc.). The ``type`` is
            swapped per task to ``classifiers[task_type]``; PCA is never applied
            here (encoders ship PCA-50 baked; stat_simple/fe_xgboost have none).
        seed: random_state for the probe.

    Returns:
        ``{task: TaskPrediction}`` — y_pred is class probability for binary tasks,
        else the point prediction.
    """
    base = Path(features_dir) / name
    train = pd.read_parquet(base / "train.parquet")
    test = pd.read_parquet(base / "test.parquet")

    out: dict[str, TaskPrediction] = {}
    for task in tasks:
        ttype = get_task_type(task)
        gtr = train[train["task"] == task]
        gte = test[test["task"] == task]
        if gtr.empty or gte.empty:
            logger.warning("%s/%s: no rows (train=%d test=%d), skipping",
                           name, task, len(gtr), len(gte))
            continue

        Xtr = np.stack(gtr["features"].to_numpy())
        Xte = np.stack(gte["features"].to_numpy())
        ytr = gtr["label"].to_numpy()
        yte = gte["label"].to_numpy()
        if ttype in ("binary", "multiclass", "ordinal"):
            ytr, yte = ytr.astype(int), yte.astype(int)

        cfg = replace(base_config, type=classifiers[ttype], pca_n_components=None)
        model = create_model(cfg, random_state=seed, task_type=ttype)
        model.fit(Xtr, ytr)

        if ttype == "binary":
            y_pred = model.predict_proba(Xte)[:, 1]
        else:  # ordinal / multiclass / regression → point prediction
            y_pred = model.predict(Xte)

        out[task] = TaskPrediction(
            y_true=yte, y_pred=y_pred, user_ids=gte["user_id"].to_numpy()
        )
    return out
