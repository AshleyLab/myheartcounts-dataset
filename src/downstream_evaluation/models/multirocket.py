"""MultiRocket encoder (convolutional-kernel features, per-(task, user) pooled).

MultiRocket applies random convolutional kernels (4 pooling operators over raw +
first-order-differenced series) to each daily ``(19, 24)`` segment, yielding a
49,728-d feature vector per segment. The daily-segment matrix (~1M segments ×
49,728 ≈ 200 GB) is far too large to materialize, so we chunk-transform and pool
into **per-(task, user)** running means over the segments whose label cell is
non-sentinel for that task (the embedded-temporal before-label window, IC + TC,
from the daily lookup).

Both stages run from raw data:

  - **Stage 1 (build-on-miss, CPU):** fit the kernels on the train-split segments,
    z-score with train channel stats, chunk-transform every segment, accumulate the
    per-(task, user) mean → one 49,728-d vector per (task, user). Deterministic
    (``random_state=42``).
  - **Stage 2 (eval):** ``encode_cohort(task, td)`` returns the cohort's per-user
    pooled vectors; the engine runs the uniform PCA-50 + linear probe.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
# Stage-1 helpers: norm-stat computation and z-score/zero-fill.
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


# --------------------------------------------------------------------------- #
# Stage 1 — extraction (CPU job): raw -> per-(task, user) pooled features
# --------------------------------------------------------------------------- #
def extract_multirocket(output_dir, data_dir=None, tasks=None, seed=42, chunk_size=CHUNK_SIZE):
    """Regenerate the per-(task, user) MultiRocket feature cache from raw (CPU).

    Writes ``<output_dir>/<url_escaped_task>.npz`` with ``user_ids`` + ``features``
    (n_users, 49728) float16 — the per-(task, user) pooled MultiRocket vectors.
    """
    from urllib.parse import quote

    import datasets as hf_ds
    from sktime.transformations.panel.rocket import MultiRocketMultivariate

    from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
    from downstream_evaluation.data.splits import load_split_file
    from openmhc._evaluate import _DatasetPaths, _ensure_labels_env

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)
    split_users = load_split_file(paths.splits_file)
    tasks = list(tasks) if tasks is not None else None

    logger.info("loading daily_hourly_hf: %s", paths.daily_hourly_hf)
    ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
    n = len(ds)
    seg_users = np.asarray(ds["user_id"], dtype=object).astype(str)
    seg_dates = np.asarray(ds["date"], dtype=object).astype(str)
    seg_dates = np.array([d[:10] for d in seg_dates], dtype=object)

    # daily_hourly_hf stores (19, 24) channel-first, zero-filled. Match the downstream
    # `prepare_daily_hourly_hf` step: transpose to (N, 24, 19) time-first and restore NaN
    # at missing (so the observed-only stats use the mask, not zeros).
    values = np.ascontiguousarray(np.asarray(ds["values"], dtype=np.float32).transpose(0, 2, 1))
    mask = np.ascontiguousarray(np.asarray(ds["mask"], dtype=np.float32).transpose(0, 2, 1))
    values[mask > 0.5] = np.nan

    train = {str(u) for u in split_users.get("train", [])}
    train_idx = np.where(np.array([u in train for u in seg_users]))[0]
    logger.info("segments=%d train_segments=%d", n, len(train_idx))

    means, stds = _compute_norm_stats(values, mask, train_idx)
    logger.info("norm means[:3]=%s stds[:3]=%s", means[:3], stds[:3])
    X = _zscore_zero_fill(values, mask, means, stds)
    del values, mask
    X_panel = X.transpose(0, 2, 1)  # (N, 19, T), sktime panel
    del X

    # Per-task eligible (user, date) sets from the daily lookup (non-sentinel cell =
    # in cohort + in the per-task forward window; matches valid_mask_by_task).
    lookup = str(paths.root / "processed" / LOOKUP_BY_GRANULARITY["daily"])
    provider = TaskDataProvider(lookup, split_users, granularity="daily")
    if tasks is None:
        raise ValueError("tasks must be provided for per-(task, user) pooling")
    eligible: dict[str, set] = {t: set() for t in tasks}
    for t in tasks:
        for split in ("train", "validation", "test"):
            td = provider.task_data(t, split)
            for uid, dates in zip(td.user_ids, td.dates):
                u = str(uid)
                for d in dates:
                    eligible[t].add((u, str(d)[:10]))
        logger.info("task %s: %d eligible (user,date) segments", t, len(eligible[t]))

    transformer = MultiRocketMultivariate(
        num_kernels=NUM_KERNELS, max_dilations_per_kernel=MAX_DILATIONS_PER_KERNEL,
        n_features_per_kernel=N_FEATURES_PER_KERNEL, normalise=NORMALISE, n_jobs=N_JOBS,
        random_state=seed,
    )
    logger.info("fitting MultiRocket on %d train segments...", len(train_idx))
    transformer.fit(X_panel[train_idx].astype(np.float64, copy=False))

    bl_sums: dict[str, dict[str, np.ndarray]] = {t: {} for t in tasks}
    bl_counts: dict[str, dict[str, int]] = {t: {} for t in tasks}
    from tqdm import tqdm

    for start in tqdm(range(0, n, chunk_size), desc="MultiRocket transform"):
        end = min(start + chunk_size, n)
        feats = np.asarray(
            transformer.transform(X_panel[start:end].astype(np.float64, copy=False)),
            dtype=np.float32,
        )
        cu = seg_users[start:end]
        cd = seg_dates[start:end]
        for t in tasks:
            elig = eligible[t]
            ts, tc = bl_sums[t], bl_counts[t]
            for i in range(end - start):
                key = (cu[i], cd[i])
                if key not in elig:
                    continue
                u = cu[i]
                if u in ts:
                    ts[u] += feats[i]
                    tc[u] += 1
                else:
                    ts[u] = feats[i].copy()
                    tc[u] = 1
        del feats

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        uids = sorted(bl_sums[t])
        if not uids:
            logger.warning("task %s: no pooled features", t)
            continue
        feat = np.stack([bl_sums[t][u] / max(bl_counts[t][u], 1) for u in uids]).astype(np.float16)
        np.savez(out / f"{quote(t, safe='')}.npz",
                 user_ids=np.array(uids, dtype=object), features=feat)
    logger.info("wrote per-(task, user) MultiRocket features -> %s", out)


# --------------------------------------------------------------------------- #
# Stage 2 — the MultiRocket Encoder (build-on-miss internal)
# --------------------------------------------------------------------------- #
class MultiRocket:
    """MultiRocket encoder for the engine. ``encode_cohort`` returns the cohort's
    per-user pooled 49,728-d features (built on a cache miss, loaded on a hit); the
    engine then runs the uniform PCA-50 + probe.
    """

    name = "multirocket"
    input_granularity = "daily"  # per-user cohort from the daily lookup
    needs_segments = False  # consumes its own build-on-miss feature cache

    def __init__(self, data_dir=None, cache_dir=None, tasks=None, seed=42):
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._tasks = list(tasks) if tasks is not None else None
        self.seed = seed
        self._by_task: dict[str, dict[str, np.ndarray]] = {}

    def _resolve_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        return Path("results") / "multirocket_features" / "from_raw"

    def _ensure_extracted(self) -> None:
        from urllib.parse import quote

        cache = self._resolve_cache_dir()
        tasks = self._tasks
        if tasks is None:
            raise ValueError("MultiRocket requires the task list (per-(task,user) pooling)")
        missing = [t for t in tasks if not (cache / f"{quote(t, safe='')}.npz").exists()]
        if missing:
            logger.info("MultiRocket cache miss (%d tasks) — extracting from raw (CPU)", len(missing))
            extract_multirocket(str(cache), data_dir=self._data_dir, tasks=tasks, seed=self.seed)

    def _load_task(self, task: str) -> dict[str, np.ndarray]:
        from urllib.parse import quote

        if task in self._by_task:
            return self._by_task[task]
        self._ensure_extracted()
        path = self._resolve_cache_dir() / f"{quote(task, safe='')}.npz"
        if not path.exists():
            logger.warning("MultiRocket features missing: %s", path)
            self._by_task[task] = {}
            return {}
        z = np.load(path, allow_pickle=True)
        uids = z["user_ids"].astype(str)
        feat = z["features"].astype(np.float32)
        self._by_task[task] = {u: feat[i] for i, u in enumerate(uids)}
        return self._by_task[task]

    def encode_cohort(self, task: str, td) -> np.ndarray:
        """Per-user pooled MultiRocket features aligned to ``td.user_ids``."""
        by_user = self._load_task(task)
        dim = len(next(iter(by_user.values()))) if by_user else 0
        X = np.zeros((len(td.user_ids), dim), dtype=np.float32)
        for i, uid in enumerate(td.user_ids):
            vec = by_user.get(str(uid))
            if vec is not None:
                X[i] = vec
        return X
