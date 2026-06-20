"""Data loading utilities for downstream evaluation.

Loads pre-computed labels from a parquet lookup table and applies filtering,
temporal clipping, and user-level splits using vectorised pandas/numpy
operations plus Arrow ``ds.select()`` for fast HF dataset subsetting.

The labels parquet is built once by ``scripts/labels/build_labels_lookup.py`` and
contains one column per task (33 total) aligned by index with the weekly HF
dataset.  This eliminates the expensive per-task label attachment that
previously dominated runtime.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import datasets as hf_ds
import numpy as np
import pandas as pd

from downstream_evaluation.data.splits import load_split_file
from labels.api import LABEL_TYPES

if TYPE_CHECKING:
    from downstream_evaluation.config import DataConfig

logger = logging.getLogger(__name__)

# Number of channels and hours per day for daily_hourly_hf datasets
_N_CHANNELS = 19
_HOURS_PER_DAY = 24


def prepare_daily_hourly_hf(ds: hf_ds.Dataset) -> hf_ds.Dataset:
    """Convert a daily_hourly_hf dataset to the downstream-pipeline format.

    daily_hourly_hf stores values/mask as (19, 24) channels-first, zero-filled
    (no NaN in values, separate mask column).  The downstream pipeline expects
    (24, 19) time-first with NaN where mask==1.

    This function applies:
      1. Transpose values/mask from (C, H) = (19, 24) → (H, C) = (24, 19)
      2. Restore NaN in values where mask == 1 (so nanmean/nanstd work correctly)

    Note: HF Dataset.map() cannot change the Array2D shape declared in the
    schema, so we rebuild the dataset from a generator with the correct
    (24, 19) features schema.

    Args:
        ds: HuggingFace Dataset from daily_hourly_hf (each row has
            ``values`` (19, 24) and ``mask`` (19, 24)).

    Returns:
        Transformed HuggingFace Dataset with ``values`` (24, 19) containing
        NaN for missing positions and ``mask`` (24, 19).
    """
    logger.info(
        "Preparing daily_hourly_hf: transposing (19,24)->(24,19) and restoring NaN (%d samples)",
        len(ds),
    )

    # Bulk-read values and mask, transpose, restore NaN
    all_vals = np.asarray(ds["values"], dtype=np.float32)  # (N, 19, 24)
    all_mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 19, 24)

    # Transpose: (N, 19, 24) → (N, 24, 19)
    all_vals = np.ascontiguousarray(all_vals.transpose(0, 2, 1))
    all_mask = np.ascontiguousarray(all_mask.transpose(0, 2, 1))

    # Restore NaN where mask == 1 (missing) so nanmean/nanstd work
    all_vals[all_mask > 0.5] = np.nan

    # Read metadata columns
    user_ids = ds["user_id"]
    dates = ds["date"]
    n_valid_hours = ds["n_valid_hours"]

    # Build new dataset with corrected schema (24, 19) instead of (19, 24)
    new_features = hf_ds.Features(
        {
            "values": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
            "mask": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
            "user_id": hf_ds.Value("string"),
            "date": hf_ds.Value("string"),
            "n_valid_hours": hf_ds.Value("int32"),
        }
    )

    new_ds = hf_ds.Dataset.from_dict(
        {
            "values": all_vals,
            "mask": all_mask,
            "user_id": user_ids,
            "date": dates,
            "n_valid_hours": n_valid_hours,
        },
        features=new_features,
    )

    logger.info(
        "Prepared: %d samples, values shape=%s",
        len(new_ds),
        np.asarray(new_ds[0]["values"]).shape,
    )
    return new_ds


class DownstreamDataLoader:
    """Unified data loading with pre-computed labels and user splits.

    Reads labels from a parquet lookup table built by
    ``scripts/labels/build_labels_lookup.py``, filters to valid samples, optionally
    applies temporal clipping, splits by user, and returns HF dataset subsets
    with the task label column attached.
    """

    def __init__(self, config: DataConfig):
        """Initialize data loader.

        Args:
            config: Data configuration with paths and split parameters.
        """
        self.config = config
        self.weekly_hf_dir = Path(config.weekly_hf_dir)
        self.task_name = config.task_name

    def load_splits(self) -> tuple[hf_ds.Dataset, hf_ds.Dataset, hf_ds.Dataset]:
        """Load data, read pre-computed labels, filter, and split by user.

        Returns:
            Tuple of (train_ds, val_ds, test_ds) HuggingFace datasets,
            each with the task label column attached.
        """
        # 1. Load full HF dataset (Arrow-backed, memory-mapped)
        logger.info(f"Loading weekly dataset from {self.weekly_hf_dir}")
        full_ds = hf_ds.load_from_disk(str(self.weekly_hf_dir))
        n_total = len(full_ds)
        logger.info(f"Loaded {n_total} weekly samples")

        # 2. Load pre-computed labels from parquet
        labels_path = Path(self.config.weekly_labels_lookup_path)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels parquet not found: {labels_path}. "
                f"Build it with: python scripts/labels/build_labels_lookup.py"
            )
        logger.info(f"Loading labels from {labels_path}")

        # Validate that required columns exist before loading
        import pyarrow.parquet as pq

        parquet_schema = pq.read_schema(labels_path)
        parquet_columns = set(parquet_schema.names)
        required_columns = {"user_id", self.task_name}
        missing_columns = required_columns - parquet_columns
        if missing_columns:
            raise ValueError(
                f"Labels parquet {labels_path} is missing columns: "
                f"{missing_columns}. Available columns: "
                f"{sorted(parquet_columns)}. "
                f"Rebuild with: python scripts/labels/build_labels_lookup.py"
            )

        labels_df = pd.read_parquet(labels_path, columns=[self.task_name, "user_id"])

        if len(labels_df) != n_total:
            raise ValueError(
                f"Labels parquet has {len(labels_df)} rows but HF dataset has "
                f"{n_total} rows. Rebuild with: python scripts/labels/build_labels_lookup.py"
            )

        labels = labels_df[self.task_name].values
        user_ids = labels_df["user_id"].values  # numpy array of str

        # 3. Build valid-label mask
        if self.task_name not in LABEL_TYPES:
            raise ValueError(
                f"Task '{self.task_name}' not found in LABEL_TYPES. "
                f"Known tasks: {sorted(LABEL_TYPES.keys())}. "
                f"Check for typos or add the task to labels/api.py."
            )
        is_continuous = LABEL_TYPES[self.task_name] == "continuous"
        missing_sentinel = -1.0 if is_continuous else -1
        valid_mask = labels != missing_sentinel
        logger.info(
            f"After filtering for {self.task_name}: "
            f"{valid_mask.sum()} / {n_total} samples have valid labels"
        )

        # 4. Apply temporal clipping mask (AND with valid_mask)
        if self.config.clip_dates_path:
            week_starts = labels_df.get("week_start")
            if week_starts is None:
                # Need week_start from parquet — reload with that column
                week_starts = pd.read_parquet(labels_path, columns=["week_start"])[
                    "week_start"
                ].values
            else:
                week_starts = week_starts.values
            clip_mask = self._temporal_clip_mask(user_ids.tolist(), week_starts.tolist())
            n_before = int(valid_mask.sum())
            valid_mask = valid_mask & clip_mask
            n_after = int(valid_mask.sum())
            n_dropped = n_before - n_after
            pct = n_dropped / n_before if n_before > 0 else 0
            logger.info(
                f"Temporal clip: {n_before} -> {n_after} samples ({n_dropped} dropped, {pct:.1%})"
            )

        # 5. Determine user splits
        valid_indices = np.where(valid_mask)[0]
        valid_user_ids = user_ids[valid_indices]
        all_users = sorted(set(valid_user_ids.tolist()))
        user_splits = self._get_user_splits(all_users)

        # 6. Partition valid indices into train/val/test using set lookups
        all_split_users = user_splits["train"] | user_splits["validation"] | user_splits["test"]
        unassigned_users = set(all_users) - all_split_users
        if unassigned_users:
            logger.warning(
                f"{len(unassigned_users)} users with valid labels are not in "
                f"any split (dropped silently). First 5: "
                f"{sorted(unassigned_users)[:5]}"
            )

        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for idx, uid in zip(valid_indices, valid_user_ids):
            if uid in user_splits["train"]:
                train_idx.append(int(idx))
            elif uid in user_splits["validation"]:
                val_idx.append(int(idx))
            elif uid in user_splits["test"]:
                test_idx.append(int(idx))

        # 7. Select subsets and attach labels.
        #    Temporarily disable HF Datasets caching so that select(),
        #    add_column(), and flatten_indices() write intermediate Arrow
        #    files to a system temp dir (typically /tmp) instead of the
        #    dataset's NFS directory.  This prevents disk quota exhaustion
        #    from multi-GB cache files accumulating across 92 task runs.
        caching_was_enabled = hf_ds.is_caching_enabled()
        hf_ds.disable_caching()

        split_datasets = []
        try:
            for split_idx, split_labels_idx in [
                (train_idx, train_idx),
                (val_idx, val_idx),
                (test_idx, test_idx),
            ]:
                ds = full_ds.select(split_idx, keep_in_memory=True)
                ds = ds.add_column(self.task_name, labels[split_labels_idx].tolist())
                ds = ds.flatten_indices(keep_in_memory=True)
                split_datasets.append(ds)
        finally:
            if caching_was_enabled:
                hf_ds.enable_caching()

        train_ds, val_ds, test_ds = split_datasets

        logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} samples")

        return train_ds, val_ds, test_ds

    def _temporal_clip_mask(self, user_ids: list[str], week_starts: list[str]) -> np.ndarray:
        """Build a boolean mask: True if week_start <= label_date for that user.

        Uses clip_dates.json: {task_name: {user_id: label_date_iso}}.
        Users without clip dates keep all their weeks.

        Args:
            user_ids: List of user ID strings (length N).
            week_starts: List of week_start strings (length N).

        Returns:
            Boolean mask of shape (N,).
        """
        clip_path = Path(self.config.clip_dates_path)
        if not clip_path.exists():
            raise FileNotFoundError(
                f"clip_dates_path not found: {clip_path}. Either create it, "
                f"fix the path, or set clip_dates_path=None to disable "
                f"temporal clipping."
            )

        with clip_path.open("r") as f:
            all_clip_dates = json.load(f)

        task_clip_dates = all_clip_dates.get(self.task_name, {})
        if not task_clip_dates:
            logger.warning(f"No clip dates for task '{self.task_name}', skipping temporal clip")
            return np.ones(len(user_ids), dtype=bool)

        # Pre-convert clip dates to int64 nanoseconds for fast comparison
        clip_ns = {
            uid: int(pd.Timestamp(date_str).value) for uid, date_str in task_clip_dates.items()
        }

        # Vectorised timestamp conversion
        week_ns = pd.to_datetime(week_starts).astype(np.int64).values

        mask = np.ones(len(user_ids), dtype=bool)
        for i in range(len(user_ids)):
            limit_ns = clip_ns.get(user_ids[i])
            if limit_ns is not None:
                if int(week_ns[i]) > limit_ns:
                    mask[i] = False
        return mask

    def _get_user_splits(self, user_ids: list[str]) -> dict[str, set[str]]:
        """Load or generate user splits.

        Uses the same functions as baseline_datamodule.py to ensure consistency.

        Raises:
            FileNotFoundError: If split_file is set but the file does not exist.
        """
        split_path = Path(self.config.split_file)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. "
                f"Please provide a valid split file via data.split_file in the config."
            )
        logger.info(f"Loading splits from {split_path}")
        return load_split_file(split_path)

    @staticmethod
    def _purge_cache_files(dataset_dir: str) -> None:
        """Remove cache-*.arrow and tmp* files from the dataset directory.

        HuggingFace Datasets writes intermediate Arrow cache files during
        select() and add_column() operations.  Even with keep_in_memory=True,
        add_column() may write cache files that cleanup_cache_files() won't
        remove (because the dataset object still references them).  This
        method force-removes them after the split datasets are fully built
        in memory.

        Args:
            dataset_dir: Path to the HF dataset directory.
        """
        patterns = [
            os.path.join(dataset_dir, "cache-*.arrow"),
            os.path.join(dataset_dir, "tmp*"),
        ]
        removed = 0
        freed_bytes = 0
        for pattern in patterns:
            for fpath in glob.glob(pattern):
                try:
                    size = os.path.getsize(fpath)
                    os.remove(fpath)
                    removed += 1
                    freed_bytes += size
                except OSError:
                    pass
        if removed:
            logger.info(
                f"Purged {removed} cache files ({freed_bytes / 1e9:.1f} GB) from {dataset_dir}"
            )
