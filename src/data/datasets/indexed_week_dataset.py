"""On-the-fly weekly dataset that builds (168, 38) tensors from daily_hourly_hf + window index.

This is a drop-in replacement for loading a pre-materialised ``weekly_hf``
dataset.  Instead of reading from a pre-built weekly Arrow dataset, it:

1. Loads the lightweight ``daily_hourly_hf`` dataset (memory-mapped Arrow,
   ~16 GB, each row is one user-day at hourly resolution: (24, 19)).
2. Reads a window index (Parquet) that maps each weekly sample to a list of
   daily row indices.
3. In ``__getitem__``, concatenates the daily rows into a (168, 19) raw values
   tensor plus a (168, 19) missingness mask — the *same* format that the
   existing ``PairWeekDatasetHFRaw`` wrapper expects.

Key design decisions:
- **No resampling needed**: ``daily_hourly_hf`` is already at hourly resolution.
  The minute→hourly resampling (``resample_day_to_hourly``) was done once when
  the ``daily_hourly_hf`` was built.
- **Zero-copy reads**: HuggingFace Arrow datasets are memory-mapped, so reading
  7 rows per ``__getitem__`` is essentially 7 pointer dereferences + memcpy.
- **Flexible windowing**: different window indices (stride, window_size,
  min_data_days) can be swapped without rebuilding the underlying data.
- **Compatible output**: produces a dict with the same keys as ``weekly_hf``
  (``values``, ``mask``, ``user_id``, ``week_start``, ``n_data_days``,
  ``n_valid_days``, ``n_valid_hours``), so ``PairWeekDatasetHFRaw`` works
  unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import datasets as hf_ds
import numpy as np
import pandas as pd

from data.processing.build_window_index import load_window_index

logger = logging.getLogger(__name__)

N_CHANNELS = 19
HOURS_PER_DAY = 24


class IndexedWeekDataset:
    """Virtual weekly dataset backed by daily_hourly_hf + window index.

    This class exposes a dict-style interface that is compatible with what
    ``AppleContrastiveWeeklyDataModule.setup()`` expects from a loaded
    ``weekly_hf`` dataset.  Specifically, each element is a dict with keys:

    - ``values``: np.ndarray (window_size * 24, 19) — raw hourly sensor values
      (NaN filled with 0.0).
    - ``mask``: np.ndarray (window_size * 24, 19) — binary missingness mask
      (1=missing, 0=observed).
    - ``user_id``: str
    - ``week_start``: str (ISO date)
    - ``n_data_days``: int
    - ``n_valid_days``: int — days with >= 12 valid hours (same definition as
      weekly_hf builder).
    - ``n_valid_hours``: int — total hours with any observed channel.

    This means it can be wrapped by ``PairWeekDatasetHFRaw`` without changes.

    Args:
        daily_hourly_ds: A HuggingFace ``Dataset`` of daily hourly records.
            Expected columns: ``values`` (24, 19), ``user_id`` (str),
            ``date`` (str).
        window_index: DataFrame from ``build_window_index`` or
            ``load_window_index``.  Must have columns ``user_id``,
            ``window_start``, ``n_data_days``, ``row_indices``.
        window_size: Number of days per window.  Must match the
            ``window_size`` used to build the index.
    """

    # Columns that the class exposes (used by column_names property)
    _COLUMN_NAMES = [
        "values",
        "mask",
        "user_id",
        "week_start",
        "n_data_days",
        "n_valid_days",
        "n_valid_hours",
    ]

    def __init__(
        self,
        daily_hourly_ds: hf_ds.Dataset,
        window_index: pd.DataFrame,
        window_size: int = 7,
    ):
        """Initialize from a daily hourly dataset and a window index DataFrame."""
        self.daily_ds = daily_hourly_ds
        self.window_index = window_index.reset_index(drop=True)
        self.window_size = window_size
        self._n_hours = window_size * HOURS_PER_DAY  # e.g. 168 for 7-day

        # Pre-convert row_indices to a list of lists for fast access
        self._row_indices_list: list[list[int | None]] = list(
            self.window_index["row_indices"]
        )

        logger.info(
            "IndexedWeekDataset: %d windows, %d daily rows, window_size=%d",
            len(self.window_index),
            len(self.daily_ds),
            self.window_size,
        )

    @staticmethod
    def _normalise_week_start(ws: str) -> str:
        """Normalise window_start to 'YYYY-MM-DD' for labels_lookup compat."""
        # Strip time component if present (e.g. "2022-06-23 00:00:00" → "2022-06-23")
        return ws[:10]

    # ------------------------------------------------------------------
    # Dict-like interface expected by PairWeekDatasetHFRaw / DataModule
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.window_index)

    def __iter__(self):
        """Iterate over all windows (used by PairWeekDatasetHFRaw._build_user_index)."""
        for i in range(len(self)):
            yield self._fast_metadata(i)

    def _fast_metadata(self, idx: int) -> dict:
        """Return lightweight metadata for a window (no data reading).

        This is used during iteration for building the user index — we only
        need ``user_id`` and ``week_start``, not the full values/mask tensors.
        """
        row = self.window_index.iloc[idx]
        return {
            "user_id": str(row["user_id"]),
            "week_start": self._normalise_week_start(str(row["window_start"])),
            "n_data_days": int(row["n_data_days"]),
        }

    def __getitem__(self, idx: int | slice | str) -> dict | list:
        """Build weekly sample(s) on-the-fly from daily rows.

        Supports three access patterns for HuggingFace Dataset compatibility:
        1. ``dataset[int]`` — returns one sample dict
        2. ``dataset[slice]`` — returns a batched dict with array values
        3. ``dataset["column_name"]`` — returns all values for that column

        Args:
            idx: Window index (int), slice, or column name (str).

        Returns:
            Dict matching the weekly_hf schema. For slices, values are batched
            numpy arrays (e.g., ``values`` shape ``(B, 168, 19)``).
            For string keys, returns a list of all values for that column.
        """
        if isinstance(idx, str):
            return self._get_column(idx)
        if isinstance(idx, slice):
            return self._get_batch(idx)
        return self._get_single(idx)

    def _get_column(self, col: str) -> list:
        """Return all values for a column (HF Dataset batch column access).

        Supports fast paths for metadata columns (user_id, window_start,
        n_data_days) that can be read from the window index without building
        full tensors.
        """
        if col == "user_id":
            return self.window_index["user_id"].tolist()
        if col in ("week_start", "window_start"):
            return [
                self._normalise_week_start(ws)
                for ws in self.window_index["window_start"].tolist()
            ]
        if col == "n_data_days":
            return self.window_index["n_data_days"].tolist()
        if col == "split":
            if "split" in self.window_index.columns:
                return self.window_index["split"].tolist()
            return [None] * len(self)
        # For data columns (values, mask, n_valid_hours, n_valid_days),
        # we must build each sample — this is slow but correct.
        return [self._get_single(i)[col] for i in range(len(self))]

    def _get_batch(self, s: slice) -> dict:
        """Return a batch of windows as a dict of arrays (HF Dataset-compatible)."""
        indices = range(*s.indices(len(self)))
        samples = [self._get_single(i) for i in indices]
        if not samples:
            n_hours = self._n_hours
            return {
                "values": np.empty((0, n_hours, N_CHANNELS), dtype=np.float32),
                "mask": np.empty((0, n_hours, N_CHANNELS), dtype=np.float32),
                "user_id": [],
                "week_start": [],
                "n_data_days": [],
                "n_valid_days": [],
                "n_valid_hours": [],
            }
        return {
            key: (
                np.stack([s[key] for s in samples])
                if isinstance(samples[0][key], np.ndarray)
                else [s[key] for s in samples]
            )
            for key in samples[0]
        }

    def _get_single(self, idx: int) -> dict:
        row = self.window_index.iloc[idx]
        row_indices = self._row_indices_list[idx]

        values = np.zeros((self._n_hours, N_CHANNELS), dtype=np.float32)
        mask = np.ones((self._n_hours, N_CHANNELS), dtype=np.float32)  # 1=missing

        for day_offset, daily_idx in enumerate(row_indices):
            if daily_idx is None:
                continue  # Missing day stays as zeros + mask=1
            # Guard against NaN from non-JSON-encoded parquet files
            try:
                daily_idx_int = int(daily_idx)
            except (ValueError, TypeError):
                continue

            daily_row = self.daily_ds[daily_idx_int]
            day_values = np.asarray(daily_row["values"], dtype=np.float32)
            day_mask = np.asarray(daily_row["mask"], dtype=np.float32)

            # daily_hourly_hf stores (C, H) = (19, 24); we need (H, C) = (24, 19)
            if day_values.shape == (N_CHANNELS, HOURS_PER_DAY):
                day_values = day_values.T  # (19, 24) -> (24, 19)
                day_mask = day_mask.T

            start_h = day_offset * HOURS_PER_DAY
            end_h = start_h + HOURS_PER_DAY

            values[start_h:end_h] = day_values
            mask[start_h:end_h] = day_mask

        # Compute coverage metadata (matches weekly_hf builder logic)
        has_signal = np.any(mask < 0.5, axis=1)  # True if any channel observed
        n_valid_hours = int(has_signal.sum())

        # n_valid_days: days with >= 12 valid hours (same heuristic as weekly_hf)
        n_valid_days = 0
        for d in range(self.window_size):
            start_h = d * HOURS_PER_DAY
            end_h = start_h + HOURS_PER_DAY
            day_has_signal = has_signal[start_h:end_h]
            if day_has_signal.sum() >= 12:
                n_valid_days += 1

        return {
            "values": values,  # (168, 19)
            "mask": mask,  # (168, 19)
            "user_id": str(row["user_id"]),
            "week_start": self._normalise_week_start(str(row["window_start"])),
            "n_data_days": int(row["n_data_days"]),
            "n_valid_days": n_valid_days,
            "n_valid_hours": n_valid_hours,
        }

    @property
    def column_names(self) -> list[str]:
        """Return column names for compatibility with HF Dataset interface."""
        return list(self._COLUMN_NAMES)

    @property
    def user_ids(self) -> list[str]:
        """Return all user_id values (for split generation fallback)."""
        return self.window_index["user_id"].tolist()

    # ------------------------------------------------------------------
    # Filtering support (used by DataModule for user-based splitting)
    # ------------------------------------------------------------------

    def select(self, indices: list[int] | np.ndarray) -> IndexedWeekDataset:
        """Return a new IndexedWeekDataset with only the given window indices.

        This mirrors HuggingFace ``Dataset.select()`` for compatibility with
        the DataModule's user-filtering logic.

        Args:
            indices: List of window-index row positions to keep.

        Returns:
            New IndexedWeekDataset with the subset.
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        sub_df = self.window_index.iloc[indices].reset_index(drop=True)
        return IndexedWeekDataset(
            daily_hourly_ds=self.daily_ds,
            window_index=sub_df,
            window_size=self.window_size,
        )

    def filter_by_users(self, user_ids: set[str]) -> IndexedWeekDataset:
        """Return a new dataset containing only windows from the given users.

        Args:
            user_ids: Set of user_id strings to keep.

        Returns:
            New IndexedWeekDataset with the subset.
        """
        mask = self.window_index["user_id"].isin(user_ids)
        sub_df = self.window_index[mask].reset_index(drop=True)
        return IndexedWeekDataset(
            daily_hourly_ds=self.daily_ds,
            window_index=sub_df,
            window_size=self.window_size,
        )

    def filter_by_min_valid_days(
        self, min_valid_days: int
    ) -> IndexedWeekDataset:
        """Filter windows that don't meet the minimum data-days threshold.

        Uses the pre-computed ``n_data_days`` from the window index rather than
        recomputing from ``mask`` (which would require reading daily data).

        Args:
            min_valid_days: Minimum number of data days per window.

        Returns:
            New IndexedWeekDataset with the filtered subset.
        """
        mask = self.window_index["n_data_days"] >= min_valid_days
        sub_df = self.window_index[mask].reset_index(drop=True)
        return IndexedWeekDataset(
            daily_hourly_ds=self.daily_ds,
            window_index=sub_df,
            window_size=self.window_size,
        )


def load_indexed_week_dataset(
    daily_hourly_hf_dir: str | Path,
    window_index_path: str | Path,
    window_size: int = 7,
) -> IndexedWeekDataset:
    """Convenience loader: open daily_hourly_hf + window index → IndexedWeekDataset.

    Args:
        daily_hourly_hf_dir: Path to the daily_hourly_hf Arrow dataset.
        window_index_path: Path to the window index Parquet file.
        window_size: Days per window (must match the index).

    Returns:
        Ready-to-use IndexedWeekDataset.
    """
    ds = hf_ds.load_from_disk(str(daily_hourly_hf_dir))
    if isinstance(ds, hf_ds.DatasetDict):
        ds = hf_ds.concatenate_datasets(list(ds.values()))

    window_df = load_window_index(window_index_path)
    return IndexedWeekDataset(ds, window_df, window_size=window_size)
