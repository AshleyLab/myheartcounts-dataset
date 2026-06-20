"""Data loading for imputation evaluation.

Loads daily HF dataset, applies QA filters, and creates user-level splits.
Uses PyTorch DataLoader for efficient batched loading with multi-worker support.
Supports multi-day context windows for models that leverage cross-day dependencies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import TYPE_CHECKING

import datasets as hf_ds
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.filters.daily_filters import (
    Filter,
    LowChannelVarianceFilter,
    WearTimeFilter,
    apply_filters,
)
from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS
from data.transforms.nan_transforms import ZeroToNaNTransform

from .splits import load_split_file

if TYPE_CHECKING:
    from imputation_evaluation.config import DataConfig

logger = logging.getLogger(__name__)


def build_multiday_windows(
    hf_dataset: hf_ds.Dataset,
    indices: list[int],
    n_days: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Group daily samples into non-overlapping multi-day windows.

    For each user, sorts days chronologically (ignoring calendar gaps),
    chunks into groups of n_days. Last incomplete group is left-padded
    with -1 sentinels (NaN in dataset).

    Args:
        hf_dataset: The HuggingFace daily dataset with user_id and date columns.
        indices: HF dataset indices for this split.
        n_days: Number of days per window.

    Returns:
        Tuple ``(windows, day_offsets)`` where each list has the same length:
          - ``windows[i]`` is a list of ``n_days`` split-local indices
            (``-1`` = padding).
          - ``day_offsets[i]`` is a list of ``n_days`` calendar-day deltas from
            the first non-padded day in the window (``-1`` = padding). Used by
            models with RoPE day embeddings to encode real calendar gaps.
    """
    # Batch-extract user_id and date using numpy for efficiency
    all_user_ids = np.array(hf_dataset["user_id"])
    all_dates = np.array(hf_dataset["date"])

    hf_indices = np.array(indices)
    user_ids = all_user_ids[hf_indices]
    dates = all_dates[hf_indices]

    # Group split-local indices by user_id (carry the date string for offsets)
    user_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split_local_idx, (user_id, date) in enumerate(zip(user_ids, dates)):
        user_groups[user_id].append((date, split_local_idx))

    windows: list[list[int]] = []
    day_offsets_all: list[list[int]] = []
    for user_id in sorted(user_groups.keys()):
        # Sort by date (chronological order, ignoring calendar gaps)
        days = sorted(user_groups[user_id], key=lambda x: x[0])
        split_local_indices = [d[1] for d in days]
        sorted_dates = [d[0] for d in days]

        # Chunk into non-overlapping windows of n_days
        for i in range(0, len(split_local_indices), n_days):
            window = split_local_indices[i : i + n_days]
            chunk_dates = sorted_dates[i : i + n_days]
            # Calendar-day offsets relative to first day in the chunk
            first = _date.fromisoformat(chunk_dates[0])
            offsets = [(_date.fromisoformat(d) - first).days for d in chunk_dates]
            # Left-pad last incomplete group with -1
            if len(window) < n_days:
                pad = n_days - len(window)
                window = [-1] * pad + window
                offsets = [-1] * pad + offsets
            windows.append(window)
            day_offsets_all.append(offsets)

    return windows, day_offsets_all


@dataclass
class DailySample:
    """A single daily sample for imputation evaluation.

    Attributes:
        data: Sensor values of shape (19, 1440), NaN for missing values.
        original_mask: Binary mask of shape (19, 1440), 1=valid, 0=NaN/missing.
        user_id: User identifier.
        date: Date string (YYYY-MM-DD).
    """

    data: np.ndarray
    original_mask: np.ndarray
    user_id: str
    date: str


class SubsetWithOriginalIndices(Dataset):
    """Subset of a dataset that also returns the original split-local index.

    When pre-generated masks only cover a fraction of samples, this avoids
    loading and processing unmasked samples. The original index is returned
    as a third element so the evaluator can map back to mask lookups.
    """

    def __init__(self, dataset: Dataset, subset_indices: list[int]):
        """Initialize the subset.

        Args:
            dataset: The underlying dataset.
            subset_indices: Sorted list of original split-local indices to include.
        """
        self._dataset = dataset
        self._subset_indices = subset_indices

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self._subset_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (data, mask, original_index) for the given subset position."""
        original_idx = self._subset_indices[idx]
        data, mask = self._dataset[original_idx]
        return data, mask, original_idx


class ImputationDataset(Dataset):
    """PyTorch Dataset for imputation evaluation.

    Returns (data, original_mask) tuples of shape (19, 1440).
    Uses memory-mapped HuggingFace dataset for efficient access.
    """

    def __init__(
        self,
        hf_dataset: hf_ds.Dataset,
        indices: list[int],
        zero_to_nan_transform: ZeroToNaNTransform | None,
    ):
        """Initialize the dataset.

        Args:
            hf_dataset: The HuggingFace dataset (memory-mapped).
            indices: Indices into the dataset for this split.
            zero_to_nan_transform: Optional preprocessing transform.
        """
        self._dataset = hf_dataset
        self._indices = indices
        self._zero_to_nan_transform = zero_to_nan_transform

    def __len__(self) -> int:
        """Return number of samples in this dataset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        Args:
            idx: Index into this dataset (not the underlying HF dataset).

        Returns:
            Tuple of (data, original_mask) tensors with shape (19, 1440).
        """
        dataset_idx = self._indices[idx]
        row = self._dataset[dataset_idx]

        values = torch.as_tensor(row["values"], dtype=torch.float32)

        if self._zero_to_nan_transform is not None:
            values = self._zero_to_nan_transform(values)

        original_mask = (~torch.isnan(values)).float()

        return values, original_mask


class MultiDayImputationDataset(Dataset):
    """PyTorch Dataset for multi-day imputation evaluation.

    Returns (data, original_mask) tuples of shape (19, n_days * 1440).
    Concatenates days in each window, NaN-fills padding positions.
    """

    def __init__(
        self,
        hf_dataset: hf_ds.Dataset,
        indices: list[int],
        windows: list[list[int]],
        zero_to_nan_transform: ZeroToNaNTransform | None,
        n_days: int,
        day_offsets: list[list[int]] | None = None,
    ):
        """Initialize the multi-day dataset.

        Args:
            hf_dataset: The HuggingFace dataset (memory-mapped).
            indices: HF dataset indices for this split (split-local → HF mapping).
            windows: List of windows, each a list of n_days split-local indices (-1 = padding).
            zero_to_nan_transform: Optional preprocessing transform.
            n_days: Number of days per window.
            day_offsets: Per-window calendar offsets (parallel to ``windows``).
                ``day_offsets[i][k]`` is the calendar-day delta of slot ``k``
                from the first non-padded day in window ``i``. ``-1`` for
                padded slots. Stored as ``self._day_offsets`` for the evaluator
                to read; not returned by ``__getitem__``. ``None`` is allowed
                for backward compatibility (callers that don't need RoPE
                offsets, e.g. legacy daily MAE).
        """
        self._dataset = hf_dataset
        self._indices = indices
        self._windows = windows
        self._zero_to_nan_transform = zero_to_nan_transform
        self._n_days = n_days
        self._day_offsets = day_offsets

    def __len__(self) -> int:
        """Return number of windows in this dataset."""
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single multi-day window by index.

        Args:
            idx: Index into the windows list.

        Returns:
            Tuple of (data, original_mask) tensors with shape (19, n_days * 1440).
        """
        window = self._windows[idx]
        n_timesteps = self._n_days * 1440

        data = torch.full((19, n_timesteps), float("nan"))

        for day_offset, split_local_idx in enumerate(window):
            if split_local_idx == -1:
                continue  # Padding day: leave as NaN

            hf_idx = self._indices[split_local_idx]
            row = self._dataset[hf_idx]
            values = torch.as_tensor(row["values"], dtype=torch.float32)

            if self._zero_to_nan_transform is not None:
                values = self._zero_to_nan_transform(values)

            t_start = day_offset * 1440
            data[:, t_start : t_start + 1440] = values

        original_mask = (~torch.isnan(data)).float()
        return data, original_mask


@dataclass
class LoadedData:
    """Container for loaded data with loaders and metadata for mask generation.

    Attributes:
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        test_loader: DataLoader for test split.
        hf_dataset: The underlying HuggingFace dataset.
        split_indices: Dict mapping split name to list of global indices.
        zero_to_nan_transform: The preprocessing transform (if any).
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    hf_dataset: hf_ds.Dataset
    split_indices: dict[str, list[int]]
    zero_to_nan_transform: ZeroToNaNTransform | None
    n_days: int = 1
    window_descriptors: dict[str, list[list[int]]] | None = None
    window_day_offsets: dict[str, list[list[int]]] | None = None


class UserGroupedBatchSampler(Sampler[list[int]]):
    """Yields batches of dataset positions, grouped by user.

    Each emitted batch contains complete contiguous runs of a single
    user's dataset positions, packed up to ``batch_size`` samples.
    Multiple complete users may share one batch as long as their
    combined sample count is ``<= batch_size``. A single user with more
    samples than ``batch_size`` spans multiple contiguous batches.

    Designed for the personalized-imputer lazy state pattern: when the
    imputer holds at most one user's per-sample contributions at a
    time, user-grouped batches keep the imputer's cache warm
    throughout each user's batches without per-batch evict-and-reload
    thrashing.
    """

    def __init__(
        self,
        position_to_user: list[str],
        batch_size: int,
    ) -> None:
        """Build the batch list from a flat position → user_id list.

        Args:
            position_to_user: Dataset position ``i`` belongs to user
                ``position_to_user[i]``. Must be length ``len(dataset)``.
            batch_size: Maximum samples per batch.
        """
        # Group dataset positions by user_id, preserving each user's
        # original dataset order (so deterministic ordering is preserved
        # within a user).
        user_to_positions: dict[str, list[int]] = defaultdict(list)
        for pos, uid in enumerate(position_to_user):
            user_to_positions[uid].append(pos)

        batches: list[list[int]] = []
        current: list[int] = []
        for positions in user_to_positions.values():
            if len(positions) > batch_size:
                # User larger than batch_size: flush current, then chunk
                # the user's positions into back-to-back batches.
                if current:
                    batches.append(current)
                    current = []
                for i in range(0, len(positions), batch_size):
                    batches.append(positions[i : i + batch_size])
                continue
            if current and len(current) + len(positions) > batch_size:
                batches.append(current)
                current = []
            current.extend(positions)
        if current:
            batches.append(current)

        self._batches = batches

    def __iter__(self):
        """Yield each precomputed batch as a list of dataset positions."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)


def _build_position_to_user(
    dataset: Dataset,
    split_local_to_hf: list[int],
    hf_user_ids: list[str],
) -> list[str]:
    """Resolve each dataset position to its user_id.

    Handles both plain ``ImputationDataset`` (positions == split-local
    indices) and ``SubsetWithOriginalIndices`` (positions are subset
    positions; subset map gives split-local).
    """
    if isinstance(dataset, SubsetWithOriginalIndices):
        split_locals = dataset._subset_indices
    elif isinstance(dataset, MultiDayImputationDataset):
        # For multi-day windows, group by the user of the window's first
        # real day. Multi-day support is a TODO under user-grouped batches;
        # keep the same semantics until weekly imputers need this.
        split_locals = [next((d for d in win if d != -1), 0) for win in dataset._windows]
    else:
        split_locals = list(range(len(dataset)))
    return [hf_user_ids[split_local_to_hf[sl]] for sl in split_locals]


class ImputationDataLoader:
    """Load and split daily HF dataset for imputation evaluation.

    Applies QA filters (wear-time, variance) and creates user-level splits.
    Returns PyTorch DataLoaders for efficient batched loading.
    """

    def __init__(self, config: DataConfig):
        """Initialize the data loader.

        Args:
            config: Data configuration with paths, filters, and split settings.
        """
        self.config = config
        self._zero_to_nan_transform = (
            ZeroToNaNTransform() if config.preprocessing.zero_to_nan else None
        )

    def _build_filters(self) -> list[Filter]:
        """Build list of QA filters from config."""
        filters: list[Filter] = []

        if self.config.filters.min_wear_fraction > 0.0:
            filters.append(WearTimeFilter(self.config.filters.min_wear_fraction))

        if self.config.filters.variance_filter_enabled:
            thresholds = self.config.filters.variance_thresholds
            if thresholds is None:
                thresholds = DEFAULT_VARIANCE_THRESHOLDS
            filters.append(LowChannelVarianceFilter(thresholds))

        return filters

    def load_split_indices(
        self,
    ) -> tuple[dict[str, list[int]], list[str], list[str]]:
        """Load the HF dataset, apply QA filters, and build split indices.

        Lightweight alternative to :meth:`load_splits` that skips DataLoader/
        Dataset creation.  Useful for backfilling sample manifests on old runs.

        Returns:
            Tuple of (split_indices dict, all_user_ids list, all_dates list).
            ``all_user_ids`` and ``all_dates`` are the full columns from the
            filtered HF dataset, indexed by the global HF row index.
        """
        logger.info(f"Loading daily HF dataset from {self.config.daily_hf_dir}")
        ds = hf_ds.load_from_disk(self.config.daily_hf_dir)
        logger.info(f"Loaded {len(ds)} samples")

        filters = self._build_filters()
        if filters:
            logger.info(f"Applying {len(filters)} QA filters")
            ds = apply_filters(ds, filters)
            logger.info(f"After filtering: {len(ds)} samples")

        logger.info("Extracting user_id and date columns...")
        all_user_ids = list(ds["user_id"])
        all_dates = list(ds["date"])
        user_ids_arr = np.array(all_user_ids)
        all_users = list(np.unique(user_ids_arr))
        logger.info(f"Found {len(all_users)} unique users")

        if not self.config.split_file:
            raise ValueError(
                "DataConfig.split_file is required. The random-split fallback "
                "has been removed because it produced different user splits per "
                "run, silently breaking cross-pipeline comparisons. Point "
                "split_file at the canonical sharable_users_seed42_2026 split "
                "for your dataset version."
            )
        splits = load_split_file(Path(self.config.split_file))

        train_indices = np.where(np.isin(user_ids_arr, np.array(list(splits["train"]))))[0].tolist()
        val_indices = np.where(np.isin(user_ids_arr, np.array(list(splits["validation"]))))[
            0
        ].tolist()
        test_indices = np.where(np.isin(user_ids_arr, np.array(list(splits["test"]))))[0].tolist()

        if self.config.max_samples_per_split:
            limit = self.config.max_samples_per_split
            train_indices = train_indices[:limit]
            val_indices = val_indices[:limit]
            test_indices = test_indices[:limit]

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        logger.info(
            f"Split sizes: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
        return split_indices, all_user_ids, all_dates

    def load_splits(
        self,
        batch_size: int = 5000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> LoadedData:
        """Load and split the daily HF dataset.

        Args:
            batch_size: Number of samples per batch.
            num_workers: Number of worker processes for data loading.
            pin_memory: Pin memory for faster GPU transfer.

        Returns:
            LoadedData containing DataLoaders and metadata for mask generation.
        """
        # Load dataset (memory-mapped)
        logger.info(f"Loading daily HF dataset from {self.config.daily_hf_dir}")
        ds = hf_ds.load_from_disk(self.config.daily_hf_dir)
        logger.info(f"Loaded {len(ds)} samples")

        # Apply QA filters
        filters = self._build_filters()
        if filters:
            logger.info(f"Applying {len(filters)} QA filters")
            ds = apply_filters(ds, filters)
            logger.info(f"After filtering: {len(ds)} samples")

        # Get unique users using numpy (much faster than Python set)
        logger.info("Building user index...")
        user_ids_col = ds["user_id"]
        user_ids_arr = np.array(user_ids_col)
        all_users = list(np.unique(user_ids_arr))
        logger.info(f"Found {len(all_users)} unique users")

        # Create or load splits
        if not self.config.split_file:
            raise ValueError(
                "DataConfig.split_file is required. The random-split fallback "
                "has been removed because it produced different user splits per "
                "run, silently breaking cross-pipeline comparisons. Point "
                "split_file at the canonical sharable_users_seed42_2026 split "
                "for your dataset version."
            )
        split_path = Path(self.config.split_file)
        logger.info(f"Loading user splits from {split_path}")
        splits = load_split_file(split_path)
        # Validate split ratios match config to catch silent mismatches
        total = sum(len(v) for v in splits.values())
        if total > 0:
            actual_train = len(splits.get("train", set())) / total
            actual_val = len(splits.get("validation", set())) / total
            if abs(actual_train - self.config.train_ratio) > 0.05:
                raise ValueError(
                    f"Split file train ratio ({actual_train:.2f}) differs from configured "
                    f"train_ratio ({self.config.train_ratio}). Update the config or use a "
                    f"different split file to ensure consistency across pipelines."
                )
            if abs(actual_val - self.config.val_ratio) > 0.05:
                raise ValueError(
                    f"Split file val ratio ({actual_val:.2f}) differs from configured "
                    f"val_ratio ({self.config.val_ratio}). Update the config or use a "
                    f"different split file to ensure consistency across pipelines."
                )

        # Build index mapping using vectorized numpy operations
        logger.info("Building sample indices per split...")
        train_users = np.array(list(splits["train"]))
        val_users = np.array(list(splits["validation"]))
        test_users = np.array(list(splits["test"]))

        # Vectorized membership testing (much faster than Python loop)
        train_mask = np.isin(user_ids_arr, train_users)
        val_mask = np.isin(user_ids_arr, val_users)
        test_mask = np.isin(user_ids_arr, test_users)

        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        n_train_users = len(np.unique(user_ids_arr[train_mask]))
        n_val_users = len(np.unique(user_ids_arr[val_mask]))
        n_test_users = len(np.unique(user_ids_arr[test_mask]))
        logger.info(
            "Split sizes: "
            f"train={n_train_users} users / {len(train_indices)} samples, "
            f"val={n_val_users} users / {len(val_indices)} samples, "
            f"test={n_test_users} users / {len(test_indices)} samples"
        )

        # Apply sample limit if configured (for testing)
        if self.config.max_samples_per_split:
            limit = self.config.max_samples_per_split
            train_indices = train_indices[:limit]
            val_indices = val_indices[:limit]
            test_indices = test_indices[:limit]
            logger.info(f"Limited to {limit} samples per split for testing")

        # Create PyTorch datasets (branch on n_days)
        n_days = self.config.n_days
        window_descriptors = None
        window_day_offsets = None

        if n_days > 1:
            # Build multi-day windows for each split
            logger.info(f"Building {n_days}-day windows...")
            train_windows, train_offsets = build_multiday_windows(ds, train_indices, n_days)
            val_windows, val_offsets = build_multiday_windows(ds, val_indices, n_days)
            test_windows, test_offsets = build_multiday_windows(ds, test_indices, n_days)

            window_descriptors = {
                "train": train_windows,
                "val": val_windows,
                "test": test_windows,
            }
            window_day_offsets = {
                "train": train_offsets,
                "val": val_offsets,
                "test": test_offsets,
            }

            logger.info(
                f"Windows: train={len(train_windows)}, "
                f"val={len(val_windows)}, test={len(test_windows)}"
            )

            train_dataset = MultiDayImputationDataset(
                ds,
                train_indices,
                train_windows,
                self._zero_to_nan_transform,
                n_days,
                day_offsets=train_offsets,
            )
            val_dataset = MultiDayImputationDataset(
                ds,
                val_indices,
                val_windows,
                self._zero_to_nan_transform,
                n_days,
                day_offsets=val_offsets,
            )
            test_dataset = MultiDayImputationDataset(
                ds,
                test_indices,
                test_windows,
                self._zero_to_nan_transform,
                n_days,
                day_offsets=test_offsets,
            )
        else:
            train_dataset = ImputationDataset(ds, train_indices, self._zero_to_nan_transform)
            val_dataset = ImputationDataset(ds, val_indices, self._zero_to_nan_transform)
            test_dataset = ImputationDataset(ds, test_indices, self._zero_to_nan_transform)

        # Create DataLoaders
        # Use persistent workers only if num_workers > 0
        persistent = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Order doesn't matter for statistics
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

        logger.info(
            f"Created DataLoaders with batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        )

        return LoadedData(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hf_dataset=ds,
            split_indices={
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            },
            zero_to_nan_transform=self._zero_to_nan_transform,
            n_days=n_days,
            window_descriptors=window_descriptors,
            window_day_offsets=window_day_offsets,
        )

    def create_eval_loaders(
        self,
        split_indices: dict[str, list[int]],
        hf_dataset: hf_ds.Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        window_descriptors: dict[str, list[list[int]]] | None = None,
        window_day_offsets: dict[str, list[list[int]]] | None = None,
        applicable_indices: dict[str, set[int]] | None = None,
        user_grouped_batches: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        """Create val/test DataLoaders for evaluation.

        Args:
            split_indices: Dict mapping split name to list of global indices.
            hf_dataset: The HuggingFace dataset.
            batch_size: Number of samples per batch.
            num_workers: Number of worker processes for data loading.
            pin_memory: Pin memory for faster GPU transfer.
            window_descriptors: Multi-day window descriptors (None for n_days=1).
            window_day_offsets: Per-window calendar-day offsets parallel to
                ``window_descriptors``. Forwarded to the dataset; consumed by
                the evaluator to pass real offsets to RoPE-aware models.
            applicable_indices: Optional dict mapping split name to set of split-local
                indices that have at least one mask. When provided, wraps datasets in
                SubsetWithOriginalIndices to skip unmasked samples.
            user_grouped_batches: If True, wrap val/test loaders in a
                ``UserGroupedBatchSampler`` so each batch's samples come
                from one user (or a small packing of complete users).
                Required by personalized imputers' lazy per-user state
                to avoid thrashing. Defaults to False (today's HF-order
                batches).

        Returns:
            Tuple of (val_loader, test_loader).
        """
        val_indices = split_indices["val"]
        test_indices = split_indices["test"]

        n_days = self.config.n_days
        if n_days > 1 and window_descriptors is not None:
            val_offsets = window_day_offsets["val"] if window_day_offsets else None
            test_offsets = window_day_offsets["test"] if window_day_offsets else None
            val_dataset = MultiDayImputationDataset(
                hf_dataset,
                val_indices,
                window_descriptors["val"],
                self._zero_to_nan_transform,
                n_days,
                day_offsets=val_offsets,
            )
            test_dataset = MultiDayImputationDataset(
                hf_dataset,
                test_indices,
                window_descriptors["test"],
                self._zero_to_nan_transform,
                n_days,
                day_offsets=test_offsets,
            )
        else:
            val_dataset = ImputationDataset(hf_dataset, val_indices, self._zero_to_nan_transform)
            test_dataset = ImputationDataset(hf_dataset, test_indices, self._zero_to_nan_transform)

        # Wrap in SubsetWithOriginalIndices to skip samples without any masks
        if applicable_indices is not None:
            for split_name, dataset_ref in [("val", "val_dataset"), ("test", "test_dataset")]:
                if split_name not in applicable_indices:
                    continue
                daily_indices = applicable_indices[split_name]
                if n_days > 1 and window_descriptors is not None:
                    # Convert daily indices to window indices: include a window if
                    # any of its constituent days has a mask.
                    daily_set = (
                        daily_indices if isinstance(daily_indices, set) else set(daily_indices)
                    )
                    window_indices = []
                    for w_idx, window_desc in enumerate(window_descriptors[split_name]):
                        if any(d in daily_set for d in window_desc if d != -1):
                            window_indices.append(w_idx)
                    subset_indices = window_indices
                else:
                    subset_indices = sorted(daily_indices)
                if dataset_ref == "val_dataset":
                    val_dataset = SubsetWithOriginalIndices(val_dataset, subset_indices)
                else:
                    test_dataset = SubsetWithOriginalIndices(test_dataset, subset_indices)

        # Use persistent workers only if num_workers > 0
        persistent = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

        if user_grouped_batches:
            # Pre-extract the user_id column once for both samplers.
            hf_user_ids = list(hf_dataset["user_id"])
            val_sampler = UserGroupedBatchSampler(
                _build_position_to_user(val_dataset, val_indices, hf_user_ids),
                batch_size,
            )
            test_sampler = UserGroupedBatchSampler(
                _build_position_to_user(test_dataset, test_indices, hf_user_ids),
                batch_size,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            )
            logger.info(
                "Created user-grouped eval DataLoaders: "
                "val=%d batches, test=%d batches (batch_size_cap=%d, num_workers=%d)",
                len(val_sampler),
                len(test_sampler),
                batch_size,
                num_workers,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
                drop_last=False,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
                drop_last=False,
            )

            logger.info(
                f"Created eval DataLoaders with batch_size={batch_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}"
            )

        return val_loader, test_loader
