"""TSFM (time-series foundation model) shared extraction + Encoder base.

Toto and Chronos-2 are channel-wise last-latent foundation models that share the
*entire* data path and differ only in (a) how the pretrained model is loaded and
(b) how a batch of windows is run through it. This module holds the shared,
numerically-critical extraction — continuous hourly timeline → label-date-aligned
history window → per-(split, task) HDF5 of ``(N, 19, D)`` last-latent features —
plus a :class:`TSFMEncoder` base whose ``encode_cohort`` builds those embeddings
**on a cache miss** (GPU) and channel-mean-pools them to ``(N, D)`` on a hit.

A concrete encoder (Toto, Chronos-2) supplies the two model-specific pieces:

  - ``_load_model(device) -> (handle, window_hours)``
  - ``_run_batch(handle, examples, window_hours) -> (B, 19, D) float32``

Both stages run from raw data: the per-(split, task) HDF5 is regenerated from
``daily_hourly_hf`` on each cold run rather than from a prebuilt cache. The
window is anchored to the label date at extraction time (2048 h of strictly-prior
history), so the eval's temporal-window knob does not apply — features are aligned
to the per-task cohort as produced.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import numpy as np

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24

# The 32 headline cross-sectional tasks.
FINAL_TASKS = [
    "Atrial fibrillation (Afib)", "BMI_categories", "BMI_values", "BiologicalSex", "CAD",
    "Cerebrovascular Disease", "Congenital Heart", "Diabetes", "GoSleepTime_categories", "Hdl",
    "Heart Failure or CHF", "Hypertension", "Ldl", "PH", "Peripheral/Systemic Vascular Disease",
    "SystolicBloodPressure", "TotalCholesterol", "WakeUpTime_categories", "WeightKilograms", "age",
    "blood_pressure_categories", "cardiovascular_disease", "feel_worthwhile1", "feel_worthwhile2",
    "feel_worthwhile3", "feel_worthwhile4", "framingham_risk", "satisfiedwith_life",
    "sleep_diagnosis1", "sleep_time_categories", "vigorous_act", "work",
]


def safe_task_filename(task: str) -> str:
    """Filesystem-safe filename stem for a task name (URL-escaped)."""
    return quote(task, safe="")


# --------------------------------------------------------------------------- #
# Window construction.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowExample:
    """A single model-input window and its output metadata."""

    user_id: str
    task: str
    label_date: str
    window: np.ndarray  # (19, window_hours), non-observed hours zeroed
    padding_mask: np.ndarray  # (19, window_hours) bool, True where real+finite


@dataclass
class UserTimeline:
    """Continuous hourly trajectory for one user."""

    start_date: object
    values: np.ndarray  # (n_hours, 19), NaN where unobserved
    observed_hours: np.ndarray  # (n_hours,) bool


def build_user_timeline(ds, indices: list[int], n_channels: int):
    """Build a continuous hourly timeline for one user's filtered daily rows."""
    import pandas as pd
    import torch

    from data.transforms.nan_transforms import ZeroToNaNTransform

    if not indices:
        return None
    zero_to_nan = ZeroToNaNTransform()

    rows = [ds[int(i)] for i in indices]
    rows.sort(key=lambda row: row["date"])
    first_date = pd.Timestamp(str(rows[0]["date"])[:10])
    last_date = pd.Timestamp(str(rows[-1]["date"])[:10])
    n_days = int((last_date - first_date).days) + 1
    timeline = np.full((n_days * HOURS_PER_DAY, n_channels), np.nan, dtype=np.float32)

    for row in rows:
        day = pd.Timestamp(str(row["date"])[:10])
        day_offset = int((day - first_date).days)
        start = day_offset * HOURS_PER_DAY
        values = np.asarray(row["values"], dtype=np.float32)
        if values.shape != (n_channels, HOURS_PER_DAY):
            raise ValueError(f"Unexpected values shape {values.shape} for date={row['date']}")
        values = zero_to_nan(torch.from_numpy(values)).numpy()
        timeline[start : start + HOURS_PER_DAY, :] = values.T

    observed_hours = ~np.isnan(timeline).all(axis=1)
    return UserTimeline(start_date=first_date, values=timeline, observed_hours=observed_hours)


def build_window(timeline: UserTimeline, user_id, task, label_date, window_hours, n_channels,
                 weeks_after):
    """Create a left-padded/cropped input window for one participant-task pair.

    ``weeks_after`` is the task's forward-window length in weeks (from the runner's
    temporal policy), or ``None`` to use the full history with no forward-window cap.
    """
    import pandas as pd

    # Eligible region = from the start of the participant's record up to the label
    # date plus the per-task forward window. This matches how the cohort methods'
    # lookup is built, so a participant with no pre-label data but data inside the
    # forward window is kept rather than dropped. When the forward window is disabled
    # (weeks_after is None), the whole record is eligible with no label-anchored cap.
    label_ts = pd.Timestamp(label_date).normalize()
    if weeks_after is None:
        cutoff_hours = timeline.values.shape[0]
    else:
        label_end_ts = label_ts + pd.Timedelta(weeks=weeks_after)
        cutoff_days = int((label_end_ts - timeline.start_date).days)
        cutoff_hours = cutoff_days * HOURS_PER_DAY
        if cutoff_hours <= 0:
            return None
        cutoff_hours = min(cutoff_hours, timeline.values.shape[0])
    initial_observed = timeline.observed_hours[:cutoff_hours]
    observed_positions = np.flatnonzero(initial_observed)
    if observed_positions.size == 0:
        return None

    cutoff_hours = int(observed_positions[-1]) + 1
    history = timeline.values[:cutoff_hours]

    if history.shape[0] >= window_hours:
        window_time_first = history[-window_hours:]
        padding_mask_1d = np.ones(window_hours, dtype=bool)
    else:
        pad = window_hours - history.shape[0]
        window_time_first = np.full((window_hours, n_channels), np.nan, dtype=np.float32)
        window_time_first[pad:, :] = history
        padding_mask_1d = np.zeros(window_hours, dtype=bool)
        padding_mask_1d[pad:] = True

    window_channel_first = window_time_first.T
    padding_mask = (
        np.broadcast_to(padding_mask_1d, (n_channels, window_hours))
        & np.isfinite(window_channel_first)
    ).copy()
    window = np.where(padding_mask, window_channel_first, 0.0).astype(np.float32)
    return WindowExample(
        user_id=user_id,
        task=task,
        label_date=label_ts.strftime("%Y-%m-%d"),
        window=window,
        padding_mask=padding_mask,
    )


# --------------------------------------------------------------------------- #
# Cohort / label helpers
# --------------------------------------------------------------------------- #
def _has_quality_columns(ds) -> bool:
    return {"total_nonwear_minutes", "channel_variance"}.issubset(set(ds.column_names))


def _quality_mask(ds) -> np.ndarray:
    from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS

    nonwear = np.asarray(ds["total_nonwear_minutes"], dtype=np.float64)
    keep = nonwear <= 720.0
    variances = np.asarray(ds["channel_variance"], dtype=np.float64)
    for ch, threshold in DEFAULT_VARIANCE_THRESHOLDS.items():
        ch_var = variances[:, ch]
        keep &= np.isnan(ch_var) | (ch_var >= threshold)
    return keep


def _group_indices(ds, user_to_split: dict[str, str]):
    """Group eligible HF row indices by split and user (fresh; no index-cache)."""
    user_ids = np.asarray(ds["user_id"], dtype=object)
    keep = np.array([uid in user_to_split for uid in user_ids], dtype=bool)
    assumed_prefiltered = not _has_quality_columns(ds)
    if assumed_prefiltered:
        logger.info("daily_hourly_hf has no quality columns; assuming pre-filtered rows.")
    else:
        logger.info("Applying day-level wear-time + variance filters from dataset columns.")
        keep &= _quality_mask(ds)
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx in np.where(keep)[0]:
        uid = str(user_ids[idx])
        grouped[user_to_split[uid]][uid].append(int(idx))
    return {split: dict(users) for split, users in grouped.items()}, assumed_prefiltered


def _label_timestamp(user_id, task, reference_ts):
    from labels.api import LabelTypeError, LabelValueError, get_labels

    try:
        record = get_labels(user_id, reference_ts, task)
    except (KeyError, LabelValueError, LabelTypeError, ValueError):
        return None
    if record.matched_timestamp is None:
        return None
    import pandas as pd

    return pd.Timestamp(record.matched_timestamp)


# --------------------------------------------------------------------------- #
# Streaming HDF5 writer (one file per (split, task))
# --------------------------------------------------------------------------- #
@dataclass
class _TaskWriter:
    path: Path
    n_channels: int
    attrs: dict = field(default_factory=dict)
    file: object = None
    embeddings_ds: object = None
    user_ids_ds: object = None
    dates_ds: object = None
    n_rows: int = 0
    embed_dim: int | None = None

    def append(self, embeddings, user_ids, dates):
        import h5py

        bsz = int(embeddings.shape[0])
        embed_dim = int(embeddings.shape[2])
        if self.file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = h5py.File(self.path, "w")
            self.embed_dim = embed_dim
            self.embeddings_ds = self.file.create_dataset(
                "embeddings", shape=(0, self.n_channels, embed_dim),
                maxshape=(None, self.n_channels, embed_dim), dtype="float32",
                chunks=(min(256, max(1, bsz)), self.n_channels, embed_dim),
                compression="gzip", compression_opts=4,
            )
            dt = h5py.string_dtype()
            self.user_ids_ds = self.file.create_dataset(
                "user_ids", shape=(0,), maxshape=(None,), dtype=dt, chunks=(min(1024, max(1, bsz)),)
            )
            self.dates_ds = self.file.create_dataset(
                "dates", shape=(0,), maxshape=(None,), dtype=dt, chunks=(min(1024, max(1, bsz)),)
            )
            self.file.attrs["embed_dim"] = embed_dim
            self.file.attrs["n_channels"] = self.n_channels
            for k, v in self.attrs.items():
                self.file.attrs[k] = v
        start, end = self.n_rows, self.n_rows + bsz
        self.embeddings_ds.resize((end, self.n_channels, self.embed_dim))
        self.user_ids_ds.resize((end,))
        self.dates_ds.resize((end,))
        self.embeddings_ds[start:end] = embeddings
        self.user_ids_ds[start:end] = user_ids
        self.dates_ds[start:end] = dates
        self.n_rows = end

    def close(self):
        if self.file is not None:
            self.file.attrs["n_rows"] = self.n_rows
            self.file.close()
            self.file = None


# --------------------------------------------------------------------------- #
# Encoder base (drives extraction on a cache miss, reads + pools on a hit)
# --------------------------------------------------------------------------- #
class TSFMEncoder:
    """Base for cache-based TSFM encoders (Toto, Chronos-2).

    ``encode_cohort`` ensures the requested split's per-(split, task) HDF5 exists
    (build-on-miss extraction over raw ``daily_hourly_hf``, GPU), then reads the
    task file, mean-pools across the 19 channels, and aligns to ``td.user_ids``.
    Subclasses override ``_load_model`` and ``_run_batch``.
    """

    name = "tsfm"  # overridden
    input_granularity = "daily"  # label-aligned per-(user, task) cohort = daily lookup
    needs_segments = False  # consumes its own build-on-miss HDF5 cache
    pooling_label = "last_latent"  # overridden (provenance only)

    def __init__(self, data_dir: str | None = None, cache_dir: str | None = None,
                 batch_size: int = 32, seed: int = 42):
        """Configure data root, embedding cache dir, batch size, and seed."""
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self.seed = seed
        self._built: set[str] = set()  # splits whose HDF5s are present
        self._ds = None  # cached daily_hourly_hf
        self._grouped = None  # {split: {user: [row idx]}}
        self._handle = None  # loaded model
        self._window_hours: int | None = None
        self._task_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._temporal = None  # forward-window policy, injected by run_eval

    def set_temporal_window(self, temporal) -> None:
        """Receive the runner's forward-window policy (``run_eval`` injects this)."""
        self._temporal = temporal

    def _weeks_after(self, task: str) -> int | None:
        """Forward window (weeks) for ``task`` from the injected policy; ``None`` = full history."""
        if self._temporal is not None:
            return self._temporal.weeks_after(task)
        # Standalone use without the runner: fall back to the shared defaults.
        from downstream_evaluation.runner import TemporalWindowConfig

        self._temporal = TemporalWindowConfig()
        return self._temporal.weeks_after(task)

    # ----- subclass hooks ---------------------------------------------------
    def _load_model(self, device):
        """Return ``(handle, window_hours)`` for the loaded model on ``device``."""
        raise NotImplementedError

    def _run_batch(self, handle, examples, window_hours) -> np.ndarray:
        """Run a batch of ``WindowExample`` → ``(B, 19, D)`` float32."""
        raise NotImplementedError

    def _checkpoint_ref(self) -> str:
        return getattr(self, "checkpoint", "unknown")

    # ----- cache location ---------------------------------------------------
    def _resolve_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        # Full-history embeddings live in their own directory so a forward-windowed
        # cache is never silently reused for a full-history run (or vice versa). With
        # no policy injected yet, fall back to the full-history default.
        full_history = self._temporal is None or self._temporal.is_full_history
        variant = "from_raw_full_history" if full_history else "from_raw"
        return Path("results") / f"{self.name}_embeddings" / variant

    def _task_file(self, split: str, task: str) -> Path:
        return self._resolve_cache_dir() / split / f"{safe_task_filename(task)}.h5"

    # ----- stage 1: build-on-miss extraction (GPU) --------------------------
    def _ensure_loaded(self):
        """Lazily load daily_hourly_hf + the model (shared across splits)."""
        if self._ds is not None:
            return
        import datasets as hf_ds
        import torch

        from downstream_evaluation.data.splits import load_split_file
        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.resolve(self._data_dir)
        logger.info("loading daily_hourly_hf: %s", paths.daily_hourly_hf)
        self._ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
        split_users = load_split_file(paths.splits_file)
        user_to_split = {str(u): s for s, us in split_users.items() for u in us}
        self._grouped, _ = _group_indices(self._ds, user_to_split)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("%s extraction device=%s", self.name, device)
        self._handle, self._window_hours = self._load_model(device)
        logger.info("%s loaded: window_hours=%d", self.name, self._window_hours)

    def _build_split(self, split: str, tasks: list[str]) -> None:
        """Extract every (split, task) HDF5 that is missing (one timeline pass)."""
        import pandas as pd
        from tqdm import tqdm

        from data.processing.hf_config import N_CHANNELS
        from downstream_evaluation.config import LABEL_REFERENCE_DATE

        n_channels = N_CHANNELS
        reference_ts = pd.Timestamp(LABEL_REFERENCE_DATE)
        user_indices = self._grouped.get(split, {})

        pending = [t for t in tasks if not self._task_file(split, t).exists()]
        if not pending:
            return
        attrs_common = {
            "checkpoint_ref": self._checkpoint_ref(),
            "window_hours": int(self._window_hours),
            "pooling": self.pooling_label,
            "anchor": "label_date_aligned",
            "source_dataset": str(_DatasetPaths_root(self._data_dir)),
        }
        writers = {
            t: _TaskWriter(self._task_file(split, t), n_channels, {**attrs_common, "task_name": t})
            for t in pending
        }
        batches: dict[str, list] = {t: [] for t in pending}
        guard: dict[str, set] = {t: set() for t in pending}
        skips: dict[str, Counter] = {t: Counter() for t in pending}
        written: Counter = Counter()
        wa = {t: self._weeks_after(t) for t in pending}  # forward window per task

        def flush(task: str) -> None:
            batch = batches[task]
            if not batch:
                return
            emb = self._run_batch(self._handle, batch, self._window_hours)
            writers[task].append(emb, [e.user_id for e in batch], [e.label_date for e in batch])
            written[task] += int(emb.shape[0])
            batches[task] = []

        try:
            for user_id in tqdm(sorted(user_indices), desc=f"{self.name}:{split}", unit="user"):
                timeline = build_user_timeline(self._ds, user_indices[user_id], n_channels)
                if timeline is None:
                    continue
                for task in pending:
                    label_ts = _label_timestamp(user_id, task, reference_ts)
                    if label_ts is None or user_id in guard[task]:
                        continue
                    ex = build_window(
                        timeline, user_id, task, label_ts.strftime("%Y-%m-%d"),
                        self._window_hours, n_channels, wa[task],
                    )
                    if ex is None:
                        skips[task]["no_window"] += 1
                        continue
                    batches[task].append(ex)
                    guard[task].add(user_id)
                    if len(batches[task]) >= self._batch_size:
                        flush(task)
            for task in pending:
                flush(task)
        finally:
            for w in writers.values():
                w.close()
        logger.info("%s[%s] wrote %s", self.name, split, dict(written))

    def _ensure_split(self, split: str) -> None:
        if split in self._built:
            return
        self._ensure_loaded()
        self._build_split(split, FINAL_TASKS)
        self._built.add(split)

    # ----- stage 2: read + channel mean-pool --------------------------------
    def _load_task(self, split: str, task: str) -> dict[str, np.ndarray]:
        key = (split, task)
        if key in self._task_cache:
            return self._task_cache[key]
        import h5py

        path = self._task_file(split, task)
        if not path.exists():
            logger.warning("%s embeddings missing: %s", self.name, path)
            self._task_cache[key] = {}
            return {}
        with h5py.File(path, "r") as f:
            emb = f["embeddings"][:]  # (N, 19, D)
            uids = f["user_ids"][:].astype(str)
        pooled = emb.mean(axis=1)  # channel mean-pool → (N, D)
        out = {uid: pooled[i] for i, uid in enumerate(uids)}
        self._task_cache[key] = out
        return out

    def encode_cohort(self, task: str, td) -> np.ndarray:
        """Per-user channel-pooled TSFM embedding, aligned to ``td.user_ids``."""
        self._ensure_split(td.split)
        by_user = self._load_task(td.split, task)
        dim = len(next(iter(by_user.values()))) if by_user else 0
        X = np.zeros((len(td.user_ids), dim), dtype=np.float32)
        for i, uid in enumerate(td.user_ids):
            vec = by_user.get(str(uid))
            if vec is not None:
                X[i] = vec
        return X


def _DatasetPaths_root(data_dir):
    from openmhc._evaluate import _DatasetPaths

    return _DatasetPaths.resolve(data_dir).daily_hourly_hf
