"""Hybrid model (paper name: WBM) — WBM encoder primary + Linear fallback.

The reported WBM model is this hybrid, evaluated on the full cohort:
  - users with a WBM (weekly) embedding → the **SSL branch** (pooled 256-d → PCA-50
    → no-scaler linear probe, fit on SSL-users only, no demographics);
  - everyone else → the **Linear fallback** (38-d daily mean/std + demographics +
    RobustScaler probe, fit on ALL train users).

Per user, one prediction from one branch. For ranking metrics (binary AUPRC,
ordinal Spearman) each branch's predictions are percentile-ranked to [0, 1]
independently before being combined, so the two independently-trained probes sit
on a common scale; regression is left raw (Pearson r is scale-invariant).

End-to-end ``Predictor``: the engine scores ``predict`` directly (no uniform probe).
"""

from __future__ import annotations

import logging

import numpy as np

from downstream_evaluation.data.provider import TaskData
from downstream_evaluation.models.linear import Linear
from downstream_evaluation.models.wbm import DEFAULT_CHECKPOINT, WBM

logger = logging.getLogger(__name__)

_PROBE_BY_TASKTYPE = {
    "binary": "logistic_regression",
    "multiclass": "logistic_regression",
    "ordinal": "logreg_ordinal",
    "regression": "linear_regression",
}


class Hybrid:
    """WBM-primary + Linear-fallback, per-user routed, full-cohort."""

    name = "wbm"  # the reported WBM model is this hybrid
    input_granularity = "daily"  # full cohort + Linear fallback come from the daily lookup
    needs_segments = True  # the Linear branch needs the daily binder

    def __init__(
        self, data_dir: str | None = None, checkpoint: str = DEFAULT_CHECKPOINT, seed: int = 42
    ):
        self._data_dir = data_dir
        self.seed = seed
        self._linear = Linear(data_dir=data_dir, seed=seed)  # fallback branch
        self._wbm = WBM(data_dir=data_dir, checkpoint=checkpoint, seed=seed)  # SSL embeddings
        self._weekly_provider = None
        self._ssl_pca = None
        self._ssl_clf = None
        self._ttype: str | None = None

    def _weekly_td(self, task: str, split: str) -> TaskData:
        """Weekly cohort + eligible week_starts for ``task`` (for the SSL branch)."""
        if self._weekly_provider is None:
            from openmhc._evaluate import _DatasetPaths

            from downstream_evaluation.data.provider import (
                LOOKUP_BY_GRANULARITY,
                TaskDataProvider,
            )
            from downstream_evaluation.data.splits import load_split_file

            paths = _DatasetPaths.resolve(self._data_dir)
            lookup = str(paths.root / "processed" / LOOKUP_BY_GRANULARITY["weekly"])
            self._weekly_provider = TaskDataProvider(
                lookup, load_split_file(paths.splits_file), granularity="weekly"
            )
        return self._weekly_provider.task_data(task, split)

    def fit(self, task_data: TaskData) -> None:
        from sklearn.decomposition import PCA

        from downstream_evaluation.config import ClassifierConfig
        from downstream_evaluation.evaluation.metrics import get_task_type
        from downstream_evaluation.models.registry import create_model

        self._ttype = get_task_type(task_data.task)

        # Fallback: Linear (38-d + demographics + scaler) fit on ALL train users.
        self._linear.fit(task_data)

        # SSL: weekly WBM embeddings (build-on-miss) → PCA-50 → no-scaler probe,
        # fit on the SSL (weekly) train cohort only.
        wtd = self._weekly_td(task_data.task, "train")
        X_ssl = self._wbm.encode_cohort(task_data.task, wtd)  # (n_weekly, 256)
        self._ssl_pca = PCA(n_components=50, whiten=False).fit(X_ssl)
        X_ssl = self._ssl_pca.transform(X_ssl)
        y_ssl = wtd.labels
        if self._ttype in ("binary", "multiclass", "ordinal"):
            y_ssl = y_ssl.astype(int)
        cfg = ClassifierConfig(type=_PROBE_BY_TASKTYPE[self._ttype], use_scaler=False, pca_n_components=None)
        self._ssl_clf = create_model(cfg, random_state=self.seed, task_type=self._ttype)
        self._ssl_clf.fit(X_ssl, y_ssl)

    def _branch_pred(self, clf, X):
        return clf.predict_proba(X)[:, 1] if self._ttype == "binary" else clf.predict(X)

    def predict(self, task_data: TaskData) -> np.ndarray:
        """Per-user routed predictions, aligned with ``task_data.user_ids``."""
        from scipy.stats import rankdata

        daily_users = [str(u) for u in task_data.user_ids]

        # SSL-branch predictions, keyed by user (weekly cohort = SSL users).
        wtd = self._weekly_td(task_data.task, task_data.split)
        ssl_users = [str(u) for u in wtd.user_ids]
        ssl_pred = self._branch_pred(self._ssl_clf, self._ssl_pca.transform(self._wbm.encode_cohort(task_data.task, wtd)))
        ssl_by_user = dict(zip(ssl_users, ssl_pred))

        # Fallback predictions for every daily-cohort user (Linear, fit on all).
        fb_pred = np.asarray(self._linear.predict(task_data))  # aligned to daily_users
        fb_by_user = dict(zip(daily_users, fb_pred))

        ssl_set = set(ssl_users)
        is_ssl = np.array([u in ssl_set for u in daily_users])

        # Per-branch percentile-rank (ranking metrics) before combining.
        out = np.zeros(len(daily_users), dtype=np.float64)
        ssl_idx = np.where(is_ssl)[0]
        fb_idx = np.where(~is_ssl)[0]
        ssl_vals = np.array([ssl_by_user[daily_users[i]] for i in ssl_idx], dtype=np.float64)
        fb_vals = np.array([fb_by_user[daily_users[i]] for i in fb_idx], dtype=np.float64)
        if self._ttype in ("binary", "ordinal") and len(ssl_idx) and len(fb_idx):
            ssl_vals = rankdata(ssl_vals) / max(len(ssl_vals), 1)
            fb_vals = rankdata(fb_vals) / max(len(fb_vals), 1)
        out[ssl_idx] = ssl_vals
        out[fb_idx] = fb_vals
        return out
