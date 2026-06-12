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

Unified ``Method``: the engine scores ``predict``
directly — Hybrid owns both branch probes, so no uniform probe is applied.
"""

from __future__ import annotations

import logging

import numpy as np

from downstream_evaluation.data.provider import TaskData
from downstream_evaluation.models.linear import Linear
from downstream_evaluation.models.wbm import DEFAULT_CHECKPOINT, WBM

logger = logging.getLogger(__name__)


class Hybrid:
    """WBM-primary + Linear-fallback, per-user routed, full-cohort."""

    name = "wbm"  # the reported WBM model is this hybrid
    input_granularity = "daily"  # full cohort + Linear fallback come from the daily lookup
    needs_segments = True  # the Linear branch pools raw daily segments

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
        self._ctx = None  # EvalContext (task / split / cohort user_ids), injected per call

    def set_context(self, ctx) -> None:
        """Receive the per-(task, split) cohort context; the engine injects it before
        ``fit`` / ``predict``. The SSL branch keys its weekly cache by ``task`` and the
        Linear fallback needs ``user_ids`` / ``task`` — neither carried by the clean
        ``fit(data, labels, task_type)`` signature, so they arrive here."""
        self._ctx = ctx

    def set_loader(self, loader) -> None:
        """Forward the shared :class:`DataLoader` to the SSL branch: on an embedding
        cache miss its weekly windows are assembled from the same one-read store that
        serves the Linear branch's daily segments."""
        self._wbm.set_loader(loader)

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

    def fit(self, data, labels, task_type) -> None:
        from sklearn.decomposition import PCA

        from downstream_evaluation.config import PROBE_BY_TASK_TYPE, ClassifierConfig
        from downstream_evaluation.models.registry import create_model

        self._ttype = task_type

        # Fallback: Linear (38-d + demographics + scaler) fit on ALL train users.
        # Linear is a unified Method now — ``data`` is already the public (n,24,38)
        # arrays it wants, and it shares Hybrid's cohort context, so forward both.
        self._linear.set_context(self._ctx)
        self._linear.fit(data, labels, self._ttype)

        # SSL: weekly WBM embeddings (build-on-miss) → PCA-50 → no-scaler probe,
        # fit on the SSL (weekly) train cohort only.
        wtd = self._weekly_td(self._ctx.task, "train")
        X_ssl = self._wbm.encode_cohort(self._ctx.task, wtd)  # (n_weekly, 256)
        self._ssl_pca = PCA(n_components=50, whiten=False, random_state=self.seed).fit(X_ssl)
        X_ssl = self._ssl_pca.transform(X_ssl)
        y_ssl = wtd.labels
        if self._ttype in ("binary", "multiclass", "ordinal"):
            y_ssl = y_ssl.astype(int)
        cfg = ClassifierConfig(type=PROBE_BY_TASK_TYPE[self._ttype], use_scaler=False, pca_n_components=None)
        self._ssl_clf = create_model(cfg, random_state=self.seed, task_type=self._ttype)
        self._ssl_clf.fit(X_ssl, y_ssl)

    def _branch_pred(self, clf, X):
        return clf.predict_proba(X)[:, 1] if self._ttype == "binary" else clf.predict(X)

    def predict(self, data) -> np.ndarray:
        """Per-user routed predictions, aligned with ``self._ctx.user_ids``."""
        from scipy.stats import rankdata

        daily_users = [str(u) for u in self._ctx.user_ids]

        # SSL-branch predictions, keyed by user (weekly cohort = SSL users).
        wtd = self._weekly_td(self._ctx.task, self._ctx.split)
        ssl_users = [str(u) for u in wtd.user_ids]
        ssl_pred = self._branch_pred(self._ssl_clf, self._ssl_pca.transform(self._wbm.encode_cohort(self._ctx.task, wtd)))
        ssl_by_user = dict(zip(ssl_users, ssl_pred))

        # Fallback predictions for every daily-cohort user (Linear, fit on all).
        self._linear.set_context(self._ctx)
        fb_pred = np.asarray(self._linear.predict(data))  # aligned to daily_users
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
