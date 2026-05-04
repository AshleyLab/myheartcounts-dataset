"""Sample-quality filters for daily HuggingFace datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import datasets as hf_ds

from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS, MINUTES_PER_DAY
from utils.hf_cache import hf_cache_path

logger = logging.getLogger(__name__)


class Filter(ABC):
    """Base class for sample-quality filters.

    Subclasses implement ``__call__`` for use with
    ``hf_ds.Dataset.filter(..., batched=True)``.
    """

    @property
    @abstractmethod
    def required_column(self) -> str:
        """HF dataset column this filter reads."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown in logs."""

    @abstractmethod
    def __call__(self, batch: dict) -> list[bool]:
        """Return a boolean mask (True = keep) for a batched example dict."""


class WearTimeFilter(Filter):
    """Remove samples where wear-time is below a fraction of the day.

    Args:
        min_wear_fraction: Minimum fraction of the day that must be worn.
            0.0 disables filtering; 0.5 requires >= 50 % wear-time.
    """

    def __init__(self, min_wear_fraction: float = 0.5) -> None:
        """Initialize WearTimeFilter."""
        self.min_wear_fraction = min_wear_fraction
        self._max_nonwear = (1 - min_wear_fraction) * MINUTES_PER_DAY

    @property
    def required_column(self) -> str:
        """Return required column name."""
        return "total_nonwear_minutes"

    @property
    def description(self) -> str:
        """Return human-readable description."""
        return (
            f"WearTimeFilter(min_wear_fraction={self.min_wear_fraction}, "
            f"max_nonwear={self._max_nonwear:.0f} min)"
        )

    def __call__(self, batch: dict) -> list[bool]:
        """Return True for samples with sufficient wear-time."""
        return [v <= self._max_nonwear for v in batch["total_nonwear_minutes"]]


class LowChannelVarianceFilter(Filter):
    """Remove samples where any monitored channel has near-zero variance.

    Args:
        thresholds: Mapping of channel index -> minimum variance.
            Defaults to ``DEFAULT_VARIANCE_THRESHOLDS`` from ``hf_config``.
    """

    def __init__(self, thresholds: dict[int, float] | None = None) -> None:
        """Initialize LowChannelVarianceFilter."""
        self.thresholds = thresholds if thresholds is not None else DEFAULT_VARIANCE_THRESHOLDS

    @property
    def required_column(self) -> str:
        """Return required column name."""
        return "channel_variance"

    @property
    def description(self) -> str:
        """Return human-readable description."""
        return f"LowChannelVarianceFilter(thresholds={self.thresholds})"

    def __call__(self, batch: dict) -> list[bool]:
        """Return True for samples with sufficient channel variance.

        Channels with NaN variance (insufficient data) are skipped — the
        filter only rejects samples where a channel has *computable but
        too-low* variance (i.e. the device recorded data but it was flat).
        """
        import math

        keep: list[bool] = []
        for variances in batch["channel_variance"]:
            ok = True
            for ch_idx, min_var in self.thresholds.items():
                if ch_idx < len(variances):
                    v = variances[ch_idx]
                    # NaN means insufficient data (<2 valid values) — skip.
                    if math.isnan(v):
                        continue
                    if v < min_var:
                        ok = False
                        break
            keep.append(ok)
        return keep


def apply_filters(
    ds: hf_ds.Dataset,
    filters: list[Filter],
    num_proc: int = 1,
    use_cache: bool = False,
) -> hf_ds.Dataset:
    """Apply a sequence of filters to a HuggingFace dataset.

    Args:
        ds: Input dataset.
        filters: Ordered list of filters to apply.
        num_proc: Number of processes for ``Dataset.filter``.
        use_cache: Whether to load from HuggingFace cache if available.

    Returns:
        Filtered dataset.

    Raises:
        ValueError: If a required column is missing from the dataset.
    """
    for f in filters:
        col = f.required_column
        if col not in ds.column_names:
            raise ValueError(
                f"Filter {f.description} requires column '{col}' which is missing from the "
                "dataset. Rebuild the HF dataset from H5 files (set hdf5_dir in the config) "
                "to add the required columns."
            )
        n_before = len(ds)
        ds = ds.filter(
            f,
            batched=True,
            num_proc=num_proc,
            cache_file_name=hf_cache_path(f"daily_filter_{type(f).__name__}", ds),
            load_from_cache_file=use_cache,
        )
        n_after = len(ds)
        logger.info(
            f"{f.description}: {n_before} -> {n_after} samples (removed {n_before - n_after})"
        )
    return ds
