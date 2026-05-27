"""Per-channel global mode imputation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from openmhc.imputers._base import BaseImputer


class ModeImputer(BaseImputer):
    """Fill masked positions with the per-channel mode (most frequent value).

    Values are rounded to ``decimal_precision`` places before counting,
    which collapses near-duplicates on continuous channels (binary
    channels 0/1 are unaffected). Particularly natural for the binary
    channels (7-18) and the zero-heavy continuous channels (e.g. steps
    when sedentary).
    """

    name = "mode"

    def __init__(
        self,
        decimal_precision: int = 1,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(data_dir=data_dir)
        self.decimal_precision = decimal_precision
        self._channel_modes = self._compute_channel_modes()

    def _compute_channel_modes(self) -> np.ndarray:
        counters: list[Counter] = [Counter() for _ in range(self.n_channels)]
        for data, mask in self.iter_train_batches():
            valid = (mask > 0.5) & np.isfinite(data)
            for ch in range(self.n_channels):
                ch_valid = valid[:, ch, :]
                if not ch_valid.any():
                    continue
                values = data[:, ch, :][ch_valid]
                rounded = np.round(values, self.decimal_precision)
                counters[ch].update(rounded.tolist())
        modes = np.zeros(self.n_channels, dtype=np.float32)
        for ch, counter in enumerate(counters):
            if counter:
                modes[ch] = counter.most_common(1)[0][0]
        return modes

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        result = data.copy()
        for ch in range(self.n_channels):
            target = target_mask[:, ch, :] == 1
            result[:, ch, :][target] = self._channel_modes[ch]
        return result.astype(np.float32, copy=False)
