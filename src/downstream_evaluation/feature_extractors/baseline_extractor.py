"""Baseline feature extractor: per-channel mean/std over the raw sensor values.

Summarises each segment by the mean and standard deviation of its 19 raw sensor
channels (missingness skipped), giving a 38-dim vector ``[mean(19) | std(19)]``.
This keeps the baseline purely data-driven — time encoding and missingness
awareness are left to the encoder methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import datasets as hf_ds

    from downstream_evaluation.config import BaselineFeatureConfig

logger = logging.getLogger(__name__)

# Baseline uses only the 19 sensor value channels (no time features or mask).
N_SENSOR_CHANNELS = 19


def _restore_nan(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restore NaN where the mask marks an observation missing (1=missing).

    The raw dataset stores zero-filled values with a separate mask column.
    NaN-safe aggregations (``nanmean``/``nanstd``) need actual NaN to skip the
    missing observations rather than averaging in zeros.

    Args:
        vals: ``(B, T, C)`` or ``(T, C)`` zero-filled sensor values.
        mask: matching missingness mask (1=missing, 0=observed).

    Returns:
        ``vals`` with entries set to NaN where ``mask > 0.5``.
    """
    out = vals.copy()
    out[mask > 0.5] = np.nan
    return out


class BaselineFeatureExtractor:
    """Extract per-channel mean/std (38-dim) from the raw sensor segments.

    Expects a dataset whose ``values`` are time-first ``(T, 19)`` with NaN
    already restorable from the ``mask`` column — i.e. passed through
    ``prepare_daily_hourly_hf`` before extraction.
    """

    _READ_BATCH = 4096

    def __init__(self, config: BaselineFeatureConfig | None = None):
        """Initialize the extractor.

        Args:
            config: Baseline feature config. ``use_full_features=True`` (the
                stat + FFT + ARIMA 456-dim variant) is not supported here.
        """
        if config is not None and config.use_full_features:
            raise NotImplementedError(
                "BaselineFeatureExtractor supports the 38-dim mean/std baseline only; "
                "use_full_features=True is not available."
            )
        self._feature_dim = N_SENSOR_CHANNELS * 2  # 19 mean + 19 std = 38

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the extracted features (38)."""
        return self._feature_dim

    def extract_features_only(self, hf_dataset: hf_ds.Dataset) -> np.ndarray:
        """Extract per-segment mean/std features for every sample, no labels.

        Args:
            hf_dataset: dataset with ``values`` ``(T, 19)`` and matching ``mask``.

        Returns:
            ``(N, 38)`` float32 array of ``[mean(19) | std(19)]`` per segment.
        """
        n = len(hf_dataset)
        features = np.empty((n, self._feature_dim), dtype=np.float32)
        logger.info("Extracting mean/std features from %d samples...", n)

        for start in range(0, n, self._READ_BATCH):
            end = min(start + self._READ_BATCH, n)
            batch = hf_dataset[start:end]
            vals = np.asarray(batch["values"], dtype=np.float32)  # (B, T, 19)
            mask = np.asarray(batch["mask"], dtype=np.float32)  # (B, T, 19)
            vals = _restore_nan(vals, mask)

            mean = np.nanmean(vals, axis=1)  # (B, 19)
            std = np.nanstd(vals, axis=1)  # (B, 19)
            features[start:end] = np.concatenate([mean, std], axis=1)

        logger.info("Extracted mean/std features with dim=%d", features.shape[1])
        return features
