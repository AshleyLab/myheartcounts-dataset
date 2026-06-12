"""Linear baseline.

Summarizes each participant by the per-channel mean/std (38-d) of their eligible
raw daily segments, pooled per participant, with demographics appended (per-task
exclusions), then a linear probe: RobustScaler -> LogisticRegression (binary/
multiclass) / K-1 ordinal LogReg (ordinal) / OLS (regression).

Takes RAW input — there is no input normalization; the RobustScaler standardizes
the 38-d feature vector inside the probe, not the sensor data.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

_DEMO_COVARIATES = ["age", "BiologicalSex", "BMI_values"]


def _pool_mean_std(values: np.ndarray) -> np.ndarray:
    """Pool a participant's eligible segments to a 38-d ``[mean(19) | std(19)]``.

    Per segment: NaN-safe mean/std over time.
    Across segments: a plain mean (NaN from all-missing channels propagates and is
    zero-filled downstream).

    Args:
        values: ``(n_segments, T, 19)`` raw values, NaN at missing positions.

    Returns:
        ``(38,)`` pooled feature vector (may contain NaN; the caller nan-fills).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN segment/channel -> NaN (handled downstream)
        seg_mean = np.nanmean(values, axis=1)  # (n_segments, 19)
        seg_std = np.nanstd(values, axis=1)  # (n_segments, 19)
        seg_feat = np.concatenate([seg_mean, seg_std], axis=1)  # (n_segments, 38)
        return np.mean(seg_feat, axis=0)  # (38,) — plain mean over segments


class Linear:
    """Unified ``Method``: raw 38-d mean/std + demographics + a linear probe."""

    name = "linear"
    input_granularity = "daily"

    def __init__(self, data_dir: str | None = None, seed: int = 42) -> None:
        """Args:
        data_dir: dataset root (for the demographics lookup); ``MHC_DATA_DIR`` if None.
        seed: random_state for the probe.
        """
        self.seed = seed
        self._data_dir = data_dir
        self._demo_lookup: dict | None = None
        self._clf = None
        self._ttype: str | None = None
        self._ctx = None  # EvalContext (task name + cohort user_ids), injected per call

    def set_context(self, ctx) -> None:
        """Receive the cohort context; the engine injects it before ``fit`` / ``predict``.

        Demographics need the participants' ``user_ids`` (to look each one up) and the
        ``task`` name (to drop a covariate on its own task), neither of which the clean
        ``fit(data, labels, task_type)`` signature carries — so they arrive here."""
        self._ctx = ctx

    def _ensure_demo_lookup(self) -> None:
        if self._demo_lookup is not None:
            return
        import pandas as pd

        from openmhc._evaluate import _DatasetPaths

        from downstream_evaluation.demo_covariates import build_demo_user_lookup_from_labels_df

        paths = _DatasetPaths.resolve(self._data_dir)
        labels_df = pd.read_parquet(paths.daily_labels_lookup)
        self._demo_lookup = build_demo_user_lookup_from_labels_df(labels_df, _DEMO_COVARIATES)

    def _features(self, data) -> np.ndarray:
        """Per-user pooled 38-d + demographics (per-task exclusions), NaN/inf-filled.

        ``data`` is one public ``(n_segments, 24, 38)`` array per participant; channels
        0-18 are the raw values this baseline pools (the mask, 19-37, is unused)."""
        from downstream_evaluation.demo_covariates import apply_demographics

        self._ensure_demo_lookup()
        X = np.stack(
            [_pool_mean_std(np.asarray(d, dtype=np.float32)[..., :19]) for d in data]
        )  # (n_users, 38)
        X = apply_demographics(
            X, self._ctx.user_ids, self._ctx.task, self._demo_lookup, _DEMO_COVARIATES
        )
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, data, labels, task_type) -> None:
        from downstream_evaluation.config import PROBE_BY_TASK_TYPE, ClassifierConfig
        from downstream_evaluation.models.registry import create_model

        self._ttype = task_type
        X = self._features(data)
        y = labels
        if self._ttype in ("binary", "multiclass", "ordinal"):
            y = y.astype(int)
        cfg = ClassifierConfig(
            type=PROBE_BY_TASK_TYPE[self._ttype], use_scaler=True, pca_n_components=None
        )
        self._clf = create_model(cfg, random_state=self.seed, task_type=self._ttype)
        self._clf.fit(X, y)

    def predict(self, data) -> np.ndarray:
        X = self._features(data)
        if self._ttype == "binary":
            return self._clf.predict_proba(X)[:, 1]
        return self._clf.predict(X)
