"""Per-channel global mean imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer


class MeanImputer(BaseImputer):
    """Fill artificially masked positions with the per-channel global mean.

    Computes one mean per channel from the train split and uses it as the
    fill value for every masked position in that channel. Ignores
    temporal structure entirely.
    """

    name = "mean"

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        """Fit the per-channel means on the official train split."""
        super().__init__(version=version, data_dir=data_dir)
        self._channel_means = self.compute_channel_means()

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill ``target_mask == 1`` positions with the per-channel mean.

        Args:
            data: ``(N, C, T)`` float32 batch with NaN at missing cells.
            observed_mask: ``(N, C, T)``; 1 where a value is observed.
            target_mask: ``(N, C, T)``; 1 at positions to impute.

        Returns:
            A copy of ``data`` with masked positions filled; ``(N, C, T)``
            float32.
        """
        result = data.copy()
        for ch in range(self.n_channels):
            target = target_mask[:, ch, :] == 1
            result[:, ch, :][target] = self._channel_means[ch]
        return result.astype(np.float32, copy=False)
