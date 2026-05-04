"""Workout gap mask generator.

Masks HR and Active Energy channels during detected workout periods, testing
whether the model can reconstruct elevated heart rate and calorie burn
from workout activity signals.
"""

from __future__ import annotations

import numpy as np

from .base import MaskResult


class WorkoutGapMask:
    """Mask selected channels during workout periods.

    Detects workouts from workout activity channels (values > 0) and masks
    only the specified channels (default: HR + Active Energy) during those
    minutes. A sample is only applicable if at least one mask channel has
    valid (non-NaN, original_mask=1) data during workout minutes.

    Attributes:
        mask_channels: List of channel indices to mask during workouts.
        workout_channels: List of workout channel indices for detection.
    """

    def __init__(
        self,
        mask_channels: list[int] | None = None,
        workout_channels: list[int] | None = None,
    ):
        """Initialize the workout gap mask generator.

        Args:
            mask_channels: Channel indices to mask (default: [5, 6] — HR + Active Energy).
            workout_channels: Workout channel indices (default: 9-18).
        """
        self.mask_channels = mask_channels if mask_channels is not None else [5, 6]
        self.workout_channels = workout_channels or list(range(9, 19))

    @property
    def name(self) -> str:
        """Return generator name."""
        return "workout_gap"

    @property
    def is_structural(self) -> bool:
        """Return False - workout detection requires actual data values."""
        return False

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate workout gap mask.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator (not used but required by protocol).

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape
        artificial_mask = np.zeros_like(original_mask)

        # Detect workout: any workout channel > 0
        workout_detected = np.zeros(n_timesteps, dtype=bool)
        for ch in self.workout_channels:
            if ch < n_channels:
                channel_data = np.nan_to_num(data[ch], nan=0.0)
                workout_detected |= channel_data > 0

        # Check if any workout detected
        if not workout_detected.any():
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        workout_minutes = np.where(workout_detected)[0]

        # Applicability: at least one mask channel must have valid maskable data
        # during workout minutes (non-NaN and original_mask=1)
        has_maskable = False
        for ch in self.mask_channels:
            if ch < n_channels:
                valid = original_mask[ch, workout_minutes] == 1
                non_nan = ~np.isnan(data[ch, workout_minutes])
                if (valid & non_nan).any():
                    has_maskable = True
                    break

        if not has_maskable:
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        # Mask only specified channels during workout periods
        for t in workout_minutes:
            for ch in self.mask_channels:
                if ch < n_channels and original_mask[ch, t] == 1:
                    artificial_mask[ch, t] = 1

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
