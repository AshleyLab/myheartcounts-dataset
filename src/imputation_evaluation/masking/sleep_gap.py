"""Sleep gap mask generator.

Masks all channels except sleep channels during detected sleep periods,
testing whether the model can reconstruct physiological signals during sleep.
"""

from __future__ import annotations

import numpy as np

from .base import MaskResult


class SleepGapMask:
    """Mask all channels except sleep channels during sleep periods.

    Detects sleep from the asleep and inbed activity channels (values > 0)
    and masks all channels except the sleep detection channels during those minutes.

    Attributes:
        asleep_channel: Channel index for asleep activity.
        inbed_channel: Channel index for inbed activity.
    """

    def __init__(self, asleep_channel: int = 7, inbed_channel: int = 8):
        """Initialize the sleep gap mask generator.

        Args:
            asleep_channel: Channel index for asleep activity.
            inbed_channel: Channel index for inbed activity.
        """
        self.asleep_channel = asleep_channel
        self.inbed_channel = inbed_channel

    @property
    def name(self) -> str:
        """Return generator name."""
        return "sleep_gap"

    @property
    def is_structural(self) -> bool:
        """Return False - sleep detection requires actual data values."""
        return False

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate sleep gap mask.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator (not used but required by protocol).

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape
        artificial_mask = np.zeros_like(original_mask)

        # Get sleep channels (handle NaN as 0)
        asleep = np.nan_to_num(data[self.asleep_channel], nan=0.0)
        inbed = np.nan_to_num(data[self.inbed_channel], nan=0.0)

        # Detect sleep: either asleep OR inbed activity > 0
        sleep_detected = (asleep > 0) | (inbed > 0)

        # Check if any sleep detected
        if not sleep_detected.any():
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        # Mask all channels except sleep channels during sleep periods
        sleep_channels = {self.asleep_channel, self.inbed_channel}
        sleep_minutes = np.where(sleep_detected)[0]
        for t in sleep_minutes:
            for ch in range(n_channels):
                if ch in sleep_channels:
                    continue
                if original_mask[ch, t] == 1:
                    artificial_mask[ch, t] = 1

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
