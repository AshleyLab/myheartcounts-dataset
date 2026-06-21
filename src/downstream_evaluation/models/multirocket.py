"""MultiRocket baseline (convolutional-kernel features + the uniform linear probe).

MultiRocket applies random convolutional kernels (4 pooling operators over the raw +
first-order-differenced series) to each daily ``(19, 24)`` segment, yielding a
49,728-d feature vector per segment. The full daily-segment feature matrix
(~1M segments × 49,728 ≈ 200 GB) is far too large to materialize, so segments are
chunk-transformed and accumulated into a per-**user** running mean on the fly —
never cached to disk.

End-to-end ``Method``: the random kernels + per-channel train norm-stats are fit
**once** on the global train split (``random_state=42``); then each participant's
eligible segments are transformed and mean-pooled to one 49,728-d vector, on which
the uniform :class:`openmhc.LinearProbe` (PCA-50 + linear head) is fit / scored.
Under full-history a participant's eligible days are the same across tasks, so the
per-user pooled vector is **task-independent** — pooled once and reused for every task.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

N_SENSOR_CHANNELS = 19
# Matches configs/downstream_eval/multirocket.yaml (49,728-d after rounding).
NUM_KERNELS = 6250
MAX_DILATIONS_PER_KERNEL = 32
N_FEATURES_PER_KERNEL = 4
NORMALISE = False
N_JOBS = 1
CHUNK_SIZE = 50000


# --------------------------------------------------------------------------- #
# Norm-stat computation and z-score/zero-fill (train-mean imputation).
# --------------------------------------------------------------------------- #
def _compute_norm_stats(values, mask, train_idx):
    """Per-channel mean/std over observed (mask<0.5) train values — all 19 channels."""
    tv = values[train_idx]
    tm = mask[train_idx]
    observed = tm < 0.5
    vm = np.where(observed, tv, 0.0)
    counts = observed.sum(axis=(0, 1)).astype(np.float64)
    sums = vm.sum(axis=(0, 1)).astype(np.float64)
    sq = (vm**2).sum(axis=(0, 1)).astype(np.float64)
    means = np.zeros(values.shape[2], dtype=np.float64)
    stds = np.ones(values.shape[2], dtype=np.float64)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    var = np.maximum(0.0, (sq[valid] / counts[valid]) - means[valid] ** 2)
    sigma = np.sqrt(var)
    stds[valid] = np.where(sigma > 1e-6, sigma, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _zscore_zero_fill(values, mask, means, stds):
    """``(x - mean) / std`` then zero where missing (== train-mean imputation)."""
    X = (values - means) / stds
    X *= 1.0 - mask  # zero missing
    np.nan_to_num(X, copy=False, nan=0.0)
    return X


class MultiRocket:
    """Unified ``Method``: MultiRocket convolutional features + the uniform linear probe."""

    name = "multirocket"
    input_granularity = "daily"  # per-user cohort from the daily lookup
    needs_segments = False  # self-loads all segments; pools per user on the fly (no disk cache)

    def __init__(self, data_dir=None, tasks=None, seed=42):
        """``tasks`` is accepted for build-signature parity; pooling is task-independent."""
        self._data_dir = data_dir
        self.seed = seed
        self._pooled: dict | None = None  # {uid: (49728,) float32} pooled MultiRocket features
        self._probe = None
        self._ctx = None  # EvalContext (cohort user_ids), injected per call
        self._loader = None  # shared DataLoader, injected by run_eval (whole-store access)

    def set_context(self, ctx) -> None:
        """Receive the per-(task, split) cohort context; the engine injects it before
        ``fit`` / ``predict``. The clean ``Method`` signature carries no ``user_ids``,
        which MultiRocket needs to select the cohort's pooled features."""
        self._ctx = ctx

    def set_loader(self, loader) -> None:
        """Receive the shared :class:`DataLoader`; MultiRocket fits + transforms the whole
        segment store from it rather than re-reading the dataset."""
        self._loader = loader

    def _ensure_pooled(self) -> None:
        """Fit kernels + train stats on the global train split, transform every segment
        once, and accumulate a per-user 49,728-d mean. Run once; reused across tasks.

        The per-segment feature matrix (~200 GB) is never held in full — segments are
        transformed in chunks and pooled immediately.
        """
        if self._pooled is not None:
            return
        from sktime.transformations.panel.rocket import MultiRocketMultivariate
        from tqdm import tqdm

        from downstream_evaluation.data.splits import load_split_file
        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.from_root(self._data_dir)
        split_users = load_split_file(paths.splits_file)

        # Whole-history store from the shared loader (one daily_hourly_hf read across the
        # run); values/mask are already (N, 24, 19) time-first with NaN at missing.
        values, mask, seg_users = self._loader.segment_store()
        n = len(seg_users)

        train = {str(u) for u in split_users.get("train", [])}
        train_idx = np.where(np.array([u in train for u in seg_users]))[0]
        logger.info("MultiRocket segments=%d train_segments=%d", n, len(train_idx))

        means, stds = _compute_norm_stats(values, mask, train_idx)
        X = _zscore_zero_fill(values, mask, means, stds)
        del values, mask
        X_panel = X.transpose(0, 2, 1)  # (N, 19, 24) sktime panel
        del X

        transformer = MultiRocketMultivariate(
            num_kernels=NUM_KERNELS, max_dilations_per_kernel=MAX_DILATIONS_PER_KERNEL,
            n_features_per_kernel=N_FEATURES_PER_KERNEL, normalise=NORMALISE, n_jobs=N_JOBS,
            random_state=self.seed,
        )
        logger.info("MultiRocket fitting kernels on %d train segments...", len(train_idx))
        transformer.fit(X_panel[train_idx].astype(np.float64, copy=False))

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for start in tqdm(range(0, n, CHUNK_SIZE), desc="MultiRocket transform"):
            end = min(start + CHUNK_SIZE, n)
            feats = np.asarray(
                transformer.transform(X_panel[start:end].astype(np.float64, copy=False)),
                dtype=np.float32,
            )
            cu = seg_users[start:end]
            for i in range(end - start):
                u = cu[i]
                if u in sums:
                    sums[u] += feats[i]
                    counts[u] += 1
                else:
                    sums[u] = feats[i].copy()
                    counts[u] = 1
            del feats

        # Per-user mean; the float16 round-trip matches the prior cached-feature precision
        # so the pooled vectors are byte-identical to the cache-based path.
        self._pooled = {
            u: (sums[u] / max(counts[u], 1)).astype(np.float16).astype(np.float32) for u in sums
        }
        dim = next(iter(self._pooled.values())).shape[0] if self._pooled else 0
        logger.info("MultiRocket pooled per-user features: %d users, dim=%d", len(self._pooled), dim)

    def _features(self, user_ids) -> np.ndarray:
        """Per-user pooled features aligned to ``user_ids`` (missing users → zeros)."""
        dim = next(iter(self._pooled.values())).shape[0] if self._pooled else 0
        X = np.zeros((len(user_ids), dim), dtype=np.float32)
        for i, uid in enumerate(user_ids):
            vec = self._pooled.get(str(uid))
            if vec is not None:
                X[i] = vec
        return X

    def fit(self, data, labels, task_type) -> None:
        # ``data`` is unused: MultiRocket pools per user from its own self-loaded
        # segments, keyed by the cohort ``user_ids`` from set_context.
        import openmhc

        self._ensure_pooled()
        X = self._features(self._ctx.user_ids)
        self._probe = openmhc.LinearProbe(task_type, seed=self.seed).fit(X, labels)

    def predict(self, data) -> np.ndarray:
        X = self._features(self._ctx.user_ids)
        return self._probe.predict(X)
