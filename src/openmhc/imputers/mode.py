"""Per-channel global mode imputation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from openmhc._dataset import Version
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
        version: Version,
        decimal_precision: int = 1,
        data_dir: str | Path | None = None,
    ) -> None:
        """Fit the per-channel modes on the official train split.

        Args:
            version: ``"xs"`` or ``"full"`` dataset version.
            decimal_precision: Decimal places to round observed values to
                before counting frequencies.
            data_dir: Override for the dataset root.
        """
        super().__init__(version=version, data_dir=data_dir)
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
        """Fill ``target_mask == 1`` positions with the per-channel mode.

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
            result[:, ch, :][target] = self._channel_modes[ch]
        return result.astype(np.float32, copy=False)
