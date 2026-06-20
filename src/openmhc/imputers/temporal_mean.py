"""Per-(channel, minute-of-day) mean imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer


class TemporalMeanImputer(BaseImputer):
    """Fill masked positions with the per-(channel, minute) mean.

    Captures diurnal structure (e.g. circadian heart-rate / activity
    patterns) by computing one mean per channel per minute-of-day,
    folding any multi-day windows via ``t % 1440``. Falls back to the
    channel's overall mean for ``(channel, minute)`` cells with no
    training observations.
    """

    name = "temporal_mean"

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        """Fit the per-(channel, minute) means on the official train split."""
        super().__init__(version=version, data_dir=data_dir)
        self._temporal_means = self.compute_temporal_means()  # (C, 1440)

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill ``target_mask == 1`` positions with the per-(channel, minute) mean.

        Args:
            data: ``(N, C, T)`` float32 batch with NaN at missing cells.
            observed_mask: ``(N, C, T)``; 1 where a value is observed.
            target_mask: ``(N, C, T)``; 1 at positions to impute.

        Returns:
            A copy of ``data`` with masked positions filled; ``(N, C, T)``
            float32. The minute-of-day means are tiled across multi-day
            windows.
        """
        result = data.copy()
        T = data.shape[2]
        n_repeats = max(T // self.seq_len, 1)
        tiled = np.tile(self._temporal_means, (1, n_repeats))  # (C, T)
        mask = target_mask == 1
        fill = np.broadcast_to(tiled[None, :, :], result.shape)
        result[mask] = fill[mask]
        return result.astype(np.float32, copy=False)
