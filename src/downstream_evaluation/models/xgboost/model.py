"""XGBoost baseline.

A gradient-boosted-tree ``Predictor`` on hand-crafted per-participant features
(timeseries / curve-analysis / day-dynamics summaries). Tree model: features keep
NaN (XGBoost handles them natively), no scaler, no PCA, no demographics.

Self-contained two-stage from-raw flow:

  - **Stage 1 (build-on-miss, CPU):** ``extract_xgboost_features`` runs the three
    feature pipelines (timeseries → day-dynamics, which reads the timeseries daily
    checkpoints → curve-analysis) over the raw minute-level ``daily_hf`` Arrow shards,
    under a per-user 1092-day (156-week) future-data cutoff, writing the three per-user
    feature parquets.
  - **Stage 2 (eval):** the ``XGBoost`` Predictor loads that feature table (built on a
    cache miss, loaded on a hit) and fits/predicts end-to-end, routed by task type.

To skip the from-raw build, point ``features_dir`` at a prebuilt feature-table
directory to load it directly instead of rebuilding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 1000 shallow, regularized trees.
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
def _build_xgb(task_type: str, seed: int):
    """Build the bundled XGBoost estimator for a task type.

    This model is self-contained: it builds its own trees instead of routing
    through the engine's ``create_model`` (which is linear-probe-only). Estimator
    choice and hyperparameters are fixed per task type; no scaler or PCA is applied
    to this predictor path.
    """
    from xgboost import XGBClassifier, XGBRegressor

    from downstream_evaluation.models.registry import XGBOrdinalWrapper

    if task_type in ("binary", "multiclass"):
        return XGBClassifier(
            **_XGB_PARAMS,
            min_child_weight=1,
            gamma=0.0,
            eval_metric="logloss",
            random_state=seed,
        )
    if task_type == "regression":
        return XGBRegressor(**_XGB_PARAMS, objective="reg:squarederror", random_state=seed)
    if task_type == "ordinal":
        # K-1 cumulative-link wrapper (Frank & Hall). No random_state — the
        # sub-classifiers use XGBoost's default seed.
        return XGBOrdinalWrapper(params={**_XGB_PARAMS, "objective": "binary:logistic"})
    raise ValueError(f"unsupported task type: {task_type!r}")

# Per-pipeline feature tables (one row per user); joined on user_id.
_PARQUET_NAMES = [
    "pipeline_timeseries_user_features.parquet",
    "pipeline_curve_analysis_user_features.parquet",
    "pipeline_day_dynamics_user_features.parquet",
]
# Diagnostic/metadata columns to drop (they would leak coverage info).
_METADATA_PREFIXES = ("n_", "total_")

# Per user, keep daily data up to (latest valid label date + 1092 days = 156 weeks)
# and days with <=720 nonwear minutes (50% of 1440).
DEFAULT_MAX_FUTURE_DAYS = 1092
DEFAULT_MAX_NONWEAR_MINUTES = 720


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
        raise FileNotFoundError(f"no XGBoost feature tables found in {fd}")
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d, on="user_id", how="full", coalesce=True)
    drop = [c for c in merged.columns if c != "user_id" and c.startswith(_METADATA_PREFIXES)]
    return merged.drop(drop) if drop else merged


# --------------------------------------------------------------------------- #
# Stage 1 — feature build (CPU job): raw daily_hf -> 3 per-user feature parquets
# --------------------------------------------------------------------------- #
def extract_xgboost_features(
    output_dir: str,
    data_dir: str | None = None,
    max_future_days: int = DEFAULT_MAX_FUTURE_DAYS,
    max_nonwear_minutes: int = DEFAULT_MAX_NONWEAR_MINUTES,
    variance_filter: bool = True,
    force: bool = False,
) -> None:
    """Regenerate the hand-crafted per-user feature table from raw ``daily_hf`` (CPU).

    Runs the three pipelines in order — timeseries → day-dynamics (reads the timeseries
    daily checkpoints, so it must run second) → curve-analysis — and writes
    ``pipeline_{timeseries,day_dynamics,curve_analysis}_user_features.parquet`` plus
    intermediates under ``output_dir``.
    """
    from openmhc._evaluate import _DatasetPaths, _ensure_labels_env

    from downstream_evaluation.models.xgboost.pipeline_curve_analysis import (
        build_curve_analysis_features,
    )
    from downstream_evaluation.models.xgboost.pipeline_day_dynamics import (
        build_signal_processing_features,
    )
    from downstream_evaluation.models.xgboost.pipeline_timeseries import (
        build_user_features_chunked,
    )
    from downstream_evaluation.models.xgboost.preprocessing import build_cutoff_dates

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)  # build_cutoff_dates reads the Labels API
    arrow_dir = paths.daily_hf
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-user future-data cutoff = latest valid label date + max_future_days.
    cutoff_dates = build_cutoff_dates(max_future_days=max_future_days) if max_future_days > 0 else None
    suffix = f"_cutoff{max_future_days}" if max_future_days > 0 else ""
    logger.info(
        "xgboost feature build: arrow=%s out=%s nonwear<=%d cutoff_days=%d users=%s",
        arrow_dir, out, max_nonwear_minutes, max_future_days,
        len(cutoff_dates) if cutoff_dates else "all",
    )

    # 1. Timeseries — also writes the per-day checkpoints day-dynamics consumes.
    ts_out = out / "pipeline_timeseries_user_features.parquet"
    ckpt_dir = out / "timeseries_daily_chunks"
    if force or not ts_out.exists():
        build_user_features_chunked(
            arrow_dir=arrow_dir, output_path=ts_out, checkpoint_dir=ckpt_dir,
            max_nonwear_minutes=max_nonwear_minutes, variance_filter=variance_filter,
            cutoff_dates=cutoff_dates,
        )

    # 2. Day dynamics — reads the timeseries checkpoints (must run after stage 1).
    dd_out = out / "pipeline_day_dynamics_user_features.parquet"
    if force or not dd_out.exists():
        build_signal_processing_features(
            checkpoint_dir=ckpt_dir, output_path=dd_out, cutoff_dates=cutoff_dates,
        )

    # 3. Curve analysis — independent of the timeseries pipeline.
    ca_out = out / "pipeline_curve_analysis_user_features.parquet"
    if force or not ca_out.exists():
        build_curve_analysis_features(
            arrow_dir=arrow_dir, output_path=ca_out,
            checkpoint_path=out / f"curve_analysis_avg_curves{suffix}.parquet",
            max_nonwear_minutes=max_nonwear_minutes, variance_filter=variance_filter,
            cutoff_dates=cutoff_dates,
        )
    logger.info("xgboost features written -> %s", out)


# --------------------------------------------------------------------------- #
# Stage 2 — the XGBoost Predictor (build-on-miss internal)
# --------------------------------------------------------------------------- #
class XGBoost:
    """Unified ``Method``: hand-crafted per-user features + XGBoost trees.

    Features are built from raw ``daily_hf`` on a cache miss and loaded on a hit
    (``features_dir`` / the default ``results/xgboost_features/from_raw``).
    """

    name = "xgboost"
    input_granularity = "daily"  # cohort comes from the daily lookup
    needs_segments = False  # consumes its own build-on-miss feature cache, not raw segments
    predicts_from_arrays = True  # implements the unified Method contract

    def __init__(
        self,
        data_dir: str | None = None,
        seed: int = 42,
        features_dir: str | None = None,
        max_future_days: int = DEFAULT_MAX_FUTURE_DAYS,
        max_nonwear_minutes: int = DEFAULT_MAX_NONWEAR_MINUTES,
    ):
        """Args:
        data_dir: dataset root (``daily_hf`` + labels live under it); ``MHC_DATA_DIR`` if None.
        seed: random_state for the trees.
        features_dir: explicit feature-table dir to load/build into. Defaults to the
            build-on-miss cache ``results/xgboost_features/from_raw`` (point this at a
            prebuilt feature-table directory to load it instead of building).
        max_future_days / max_nonwear_minutes: from-raw build parameters (future-data
            cutoff, max non-wear minutes per day).
        """
        self.seed = seed
        self._data_dir = data_dir
        self._features_dir = features_dir
        self._max_future_days = max_future_days
        self._max_nonwear_minutes = max_nonwear_minutes
        self._index: dict[str, int] | None = None
        self._X: np.ndarray | None = None
        self._clf = None
        self._ttype: str | None = None
        self._ctx = None  # EvalContext (cohort user_ids), injected per call

    def set_context(self, ctx) -> None:
        """Receive the per-(task, split) cohort context; the engine injects it before
        ``fit`` / ``predict``. XGBoost keys its precomputed per-user feature rows by
        ``user_ids``, which the clean ``fit(data, labels, task_type)`` signature does
        not carry."""
        self._ctx = ctx

    def _resolve_features_dir(self) -> Path:
        if self._features_dir is not None:
            return Path(self._features_dir)
        return Path("results") / "xgboost_features" / "from_raw"

    def _load_features_from(self, features_dir) -> None:
        """Load the joined per-user feature table from ``features_dir`` into
        ``self._X`` / ``self._index``."""
        merged = load_handcrafted_features(features_dir)
        feature_cols = [c for c in merged.columns if c != "user_id"]
        uids = [str(u) for u in merged["user_id"].to_list()]
        X = merged.select(feature_cols).to_numpy().astype(np.float32)
        self._X = np.where(np.isinf(X), np.nan, X).astype(np.float32)  # XGBoost handles NaN
        self._index = {u: i for i, u in enumerate(uids)}
        logger.info("loaded XGBoost features: %d users x %d cols", len(uids), len(feature_cols))

    def _ensure_features(self) -> None:
        if self._index is not None:
            return
        fd = self._resolve_features_dir()
        if not all((fd / n).exists() for n in _PARQUET_NAMES):
            logger.info("xgboost feature cache miss at %s — building from raw (CPU)", fd)
            extract_xgboost_features(
                output_dir=str(fd),
                data_dir=self._data_dir,
                max_future_days=self._max_future_days,
                max_nonwear_minutes=self._max_nonwear_minutes,
            )
        self._load_features_from(fd)

    def _matrix(self, user_ids) -> np.ndarray:
        """Feature rows for ``user_ids`` (cohort users all have a feature row)."""
        return self._X[[self._index[str(u)] for u in user_ids]]

    def fit(self, data, labels, task_type) -> None:
        # ``data`` is unused: XGBoost self-serves its per-user feature rows from the
        # cache, keyed by the cohort ``user_ids`` that arrive via ``set_context``.
        self._ensure_features()
        self._ttype = task_type
        X = self._matrix(self._ctx.user_ids)
        y = labels
        if self._ttype in ("binary", "multiclass", "ordinal"):
            y = y.astype(int)
        self._clf = _build_xgb(self._ttype, self.seed)
        self._clf.fit(X, y)

    def predict(self, data) -> np.ndarray:
        X = self._matrix(self._ctx.user_ids)
        if self._ttype == "binary":
            return self._clf.predict_proba(X)[:, 1]
        return self._clf.predict(X)


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-1 XGBoost feature build (from raw, CPU).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default=None, help="dataset root (else MHC_DATA_DIR)")
    ap.add_argument("--max-future-days", type=int, default=DEFAULT_MAX_FUTURE_DAYS)
    ap.add_argument("--max-nonwear-minutes", type=int, default=DEFAULT_MAX_NONWEAR_MINUTES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    extract_xgboost_features(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_future_days=args.max_future_days,
        max_nonwear_minutes=args.max_nonwear_minutes,
        force=args.force,
    )


if __name__ == "__main__":
    _main()
