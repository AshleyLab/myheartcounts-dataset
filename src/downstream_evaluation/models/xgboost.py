"""XGBoost baseline (paper name; internal: fe_xgboost).

A gradient-boosted-tree ``Predictor`` on hand-crafted per-participant features
(timeseries / curve-analysis / day-dynamics summaries). Tree model: features keep
NaN (XGBoost handles them natively), no scaler, no PCA, no demographics.

The features are precomputed and shipped under ``<data>/features/fe_xgboost`` —
that table is the model's input cache. Recomputing them from raw is the
feature-engineering pipeline's job, not the evaluation engine's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from downstream_evaluation.data.provider import TaskData

logger = logging.getLogger(__name__)

# 1000 shallow, regularized trees — the eval's exact recipe.
_XGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
)
_XGB_BY_TASKTYPE = {
    "binary": "xgboost_classifier",
    "multiclass": "xgboost_classifier",
    "ordinal": "xgboost_ordinal",
    "regression": "xgboost_regressor",
}

# Per-pipeline feature tables (one row per user); joined on user_id.
_PARQUET_NAMES = [
    "pipeline_timeseries_user_features.parquet",
    "pipeline_curve_analysis_user_features.parquet",
    "pipeline_day_dynamics_user_features.parquet",
]
# Diagnostic/metadata columns to drop (they would leak coverage info).
_METADATA_PREFIXES = ("n_", "total_")


def load_handcrafted_features(features_dir: str | Path):
    """Full-join the per-pipeline feature tables into one per-user table.

    Drops diagnostic metadata columns (``n_*`` / ``total_*``). Column order is
    preserved across the join because XGBoost column-subsampling
    (``colsample_bytree``) selects by column index.

    Returns a polars DataFrame with a ``user_id`` column and one row per user.
    """
    import polars as pl

    fd = Path(features_dir)
    dfs = [pl.read_parquet(fd / n) for n in _PARQUET_NAMES if (fd / n).exists()]
    if not dfs:
        raise FileNotFoundError(f"no fe_xgboost feature tables found in {fd}")
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d, on="user_id", how="full", coalesce=True)
    drop = [c for c in merged.columns if c != "user_id" and c.startswith(_METADATA_PREFIXES)]
    return merged.drop(drop) if drop else merged


class XGBoost:
    """End-to-end ``Predictor``: hand-crafted per-user features + XGBoost trees."""

    name = "xgboost"
    input_granularity = "daily"  # cohort comes from the daily lookup
    needs_segments = False  # consumes its own feature cache, not raw segments

    def __init__(self, data_dir: str | None = None, seed: int = 42, features_dir: str | None = None):
        """Args:
        data_dir: dataset root (``<root>/features/fe_xgboost`` holds the tables).
        seed: random_state for the trees.
        features_dir: explicit features directory (overrides ``data_dir`` resolution).
        """
        self.seed = seed
        self._data_dir = data_dir
        self._features_dir = features_dir
        self._index: dict[str, int] | None = None
        self._X: np.ndarray | None = None
        self._clf = None
        self._ttype: str | None = None

    def _ensure_features(self) -> None:
        if self._index is not None:
            return
        from openmhc._evaluate import _DatasetPaths

        fd = self._features_dir or (
            Path(_DatasetPaths.resolve(self._data_dir).root) / "features" / "fe_xgboost"
        )
        merged = load_handcrafted_features(fd)
        feature_cols = [c for c in merged.columns if c != "user_id"]
        uids = [str(u) for u in merged["user_id"].to_list()]
        X = merged.select(feature_cols).to_numpy().astype(np.float32)
        self._X = np.where(np.isinf(X), np.nan, X).astype(np.float32)  # XGBoost handles NaN
        self._index = {u: i for i, u in enumerate(uids)}
        logger.info("loaded fe_xgboost features: %d users x %d cols", len(uids), len(feature_cols))

    def _matrix(self, user_ids) -> np.ndarray:
        """Feature rows for ``user_ids`` (cohort users all have a feature row)."""
        return self._X[[self._index[str(u)] for u in user_ids]]

    def fit(self, task_data: TaskData) -> None:
        from downstream_evaluation.config import (
            ClassifierConfig,
            XGBClassifierConfig,
            XGBOrdinalConfig,
            XGBRegressorConfig,
        )
        from downstream_evaluation.evaluation.metrics import get_task_type
        from downstream_evaluation.models.registry import create_model

        self._ensure_features()
        self._ttype = get_task_type(task_data.task)
        X = self._matrix(task_data.user_ids)
        y = task_data.labels
        if self._ttype in ("binary", "multiclass", "ordinal"):
            y = y.astype(int)
        cfg = ClassifierConfig(
            type=_XGB_BY_TASKTYPE[self._ttype],
            use_scaler=False,
            pca_n_components=None,
            xgboost_classifier=XGBClassifierConfig(**_XGB_PARAMS),
            xgboost_regressor=XGBRegressorConfig(**_XGB_PARAMS),
            xgboost_ordinal=XGBOrdinalConfig(**_XGB_PARAMS),
        )
        self._clf = create_model(cfg, random_state=self.seed, task_type=self._ttype)
        self._clf.fit(X, y)

    def predict(self, task_data: TaskData) -> np.ndarray:
        X = self._matrix(task_data.user_ids)
        if self._ttype == "binary":
            return self._clf.predict_proba(X)[:, 1]
        return self._clf.predict(X)
