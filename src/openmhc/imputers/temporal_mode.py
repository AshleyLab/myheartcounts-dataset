"""Per-(channel, minute-of-day) mode imputation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer


class TemporalModeImputer(BaseImputer):
    """Fill masked positions with the per-(channel, minute) mode.

    Like :class:`TemporalMeanImputer` but uses the most-frequent value
    at each minute. Values are rounded to ``decimal_precision`` places
    before counting. Falls back to the global per-channel mode for
    ``(channel, minute)`` cells with no training observations.
    """

    name = "temporal_mode"

    def __init__(
        self,
        version: Version,
        decimal_precision: int = 1,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(version=version, data_dir=data_dir)
        self.decimal_precision = decimal_precision
        self._temporal_modes = self._compute_temporal_modes()  # (C, 1440)

    def _compute_temporal_modes(self) -> np.ndarray:
        T = self.seq_len
        counters: list[list[Counter]] = [
            [Counter() for _ in range(T)] for _ in range(self.n_channels)
        ]
        global_counters: list[Counter] = [Counter() for _ in range(self.n_channels)]

        for data, mask in self.iter_train_batches():
            valid = (mask > 0.5) & np.isfinite(data)
            full_T = data.shape[2]
            n_folds = full_T // T
            for ch in range(self.n_channels):
                ch_valid = valid[:, ch, :]
                for k in range(n_folds):
                    s = k * T
                    day_valid = ch_valid[:, s : s + T]
                    any_valid = day_valid.any(axis=0)
                    for t_in_day in np.where(any_valid)[0]:
                        t = s + int(t_in_day)
                        vals = data[:, ch, t][ch_valid[:, t]]
                        rounded = np.round(vals, self.decimal_precision).tolist()
                        counters[ch][int(t_in_day)].update(rounded)
                        global_counters[ch].update(rounded)

        global_modes = np.zeros(self.n_channels, dtype=np.float32)
        for ch in range(self.n_channels):
            if global_counters[ch]:
                global_modes[ch] = global_counters[ch].most_common(1)[0][0]

        modes = np.zeros((self.n_channels, T), dtype=np.float32)
        for ch in range(self.n_channels):
            for t in range(T):
                if counters[ch][t]:
                    modes[ch, t] = counters[ch][t].most_common(1)[0][0]
                else:
                    modes[ch, t] = global_modes[ch]
        return modes

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        result = data.copy()
        T = data.shape[2]
        n_repeats = max(T // self.seq_len, 1)
        tiled = np.tile(self._temporal_modes, (1, n_repeats))
        mask = target_mask == 1
        fill = np.broadcast_to(tiled[None, :, :], result.shape)
        result[mask] = fill[mask]
        return result.astype(np.float32, copy=False)
