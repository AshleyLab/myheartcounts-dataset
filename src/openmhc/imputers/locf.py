"""Last Observation Carried Forward (LOCF) imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer


class LOCFImputer(BaseImputer):
    """Carry the most recent observed value forward into masked positions.

    Per ``(sample, channel)``: forward-fill from the last known anchor.
    Positions before the first anchor are back-filled (NOCB). If a
    channel has zero anchors in a sample, falls back to the global
    per-channel mean from training.
    """

    name = "locf"

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        """Fit the per-channel mean fallback on the official train split."""
        super().__init__(version=version, data_dir=data_dir)
        self._channel_means = self.compute_channel_means()

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Carry the last observed value forward into masked positions.

        Args:
            data: ``(N, C, T)`` float32 batch with NaN at missing cells.
            observed_mask: ``(N, C, T)``; 1 where a value is observed.
            target_mask: ``(N, C, T)``; 1 at positions to impute.

        Returns:
            A copy of ``data`` with masked positions filled; ``(N, C, T)``
            float32. Positions before the first anchor are back-filled;
            channels with no anchors fall back to the per-channel mean.
        """
        result = data.copy()
        N, C, T = data.shape
        for n in range(N):
            for c in range(C):
                target = target_mask[n, c, :] == 1
                if not np.any(target):
                    continue
                known = (
                    (observed_mask[n, c, :] == 1)
                    & (target_mask[n, c, :] == 0)
                    & np.isfinite(data[n, c, :])
                )
                known_idx = np.where(known)[0]
                if len(known_idx) == 0:
                    result[n, c, target] = self._channel_means[c]
                    continue

                filled = np.full(T, np.nan, dtype=np.float64)
                filled[known_idx] = data[n, c, known_idx]

                # Forward fill: propagate last known value forward.
                nan_mask = np.isnan(filled)
                idx = np.where(~nan_mask, np.arange(T), 0)
                np.maximum.accumulate(idx, out=idx)
                filled = filled[idx]

                # Backward fill leftover NaNs at the left boundary.
                nan_mask = np.isnan(filled)
                if np.any(nan_mask):
                    idx = np.where(~nan_mask, np.arange(T), T - 1)
                    idx = np.minimum.accumulate(idx[::-1])[::-1]
                    filled = filled[idx]

                result[n, c, target] = filled[target]
        return result.astype(np.float32, copy=False)
