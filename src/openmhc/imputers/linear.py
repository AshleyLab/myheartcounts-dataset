"""Linear interpolation imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openmhc.imputers._base import BaseImputer


class LinearImputer(BaseImputer):
    """Linearly interpolate between known observations along the time axis.

    Per ``(sample, channel)``, uses ``np.interp`` over the known anchor
    positions to fill masked positions. ``np.interp`` clamps outside
    the known range — so the left boundary becomes NOCB and the right
    becomes LOCF. If a channel has zero anchors in a sample, falls back
    to the global per-channel mean from training.
    """

    name = "linear"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        super().__init__(data_dir=data_dir)
        self._channel_means = self.compute_channel_means()

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
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
                query_idx = np.where(target)[0]
                if len(known_idx) == 0:
                    result[n, c, query_idx] = self._channel_means[c]
                    continue
                known_vals = data[n, c, known_idx]
                result[n, c, query_idx] = np.interp(query_idx, known_idx, known_vals)
        return result.astype(np.float32, copy=False)
