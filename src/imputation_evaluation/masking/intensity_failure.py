"""Intensity failure mask generator.

Masks HR and Active Energy channels when heart rate exceeds a threshold,
simulating sensor failure during high-intensity activity. This tests whether
the model can infer high HR from motion/context signals that remain visible.
"""

from __future__ import annotations

import numpy as np

from .base import MaskResult


class IntensityFailureMask:
    """Mask HR and energy channels during high-intensity periods.

    Detects high heart rate (above threshold) and masks only the HR and
    Active Energy channels during those minutes. Other channels (motion,
    workouts) remain visible, testing whether the model can infer high HR
    from observable context.

    Attributes:
        hr_channel: Channel index for heart rate.
        hr_threshold: Heart rate threshold in BPM (auto-converted if data in Hz).
        hr_unit: Unit mode: "auto", "bpm", or "hz".
        mask_channels: Channels to mask (default: HR + Active Energy).
    """

    # Conversion factor: 1 Hz = 60 BPM
    BPM_TO_HZ = 1.0 / 60.0

    def __init__(
        self,
        hr_channel: int = 5,
        hr_threshold: float = 160.0,
        hr_unit: str = "auto",
        mask_channels: list[int] | None = None,
    ):
        """Initialize the intensity failure mask generator.

        Args:
            hr_channel: Channel index for heart rate.
            hr_threshold: Threshold in BPM.
            hr_unit: Unit mode for threshold conversion.
            mask_channels: Channels to mask during high intensity.
        """
        self.hr_channel = hr_channel
        self.hr_threshold_bpm = hr_threshold
        self.hr_unit = hr_unit
        self.mask_channels = mask_channels or [5, 6]  # HR + Active Energy

    @property
    def name(self) -> str:
        """Return generator name."""
        return "intensity_failure"

    @property
    def is_structural(self) -> bool:
        """Return False - intensity detection requires actual HR data values."""
        return False

    def _detect_hr_unit(self, hr_data: np.ndarray) -> str:
        """Auto-detect whether HR data is in Hz or BPM.

        Heuristic: if mean valid HR < 10, assume Hz (typical HR 0.8-3 Hz).
        Otherwise assume BPM (typical HR 50-200 BPM).

        Args:
            hr_data: Heart rate values (may contain NaN).

        Returns:
            "hz" or "bpm".
        """
        valid_hr = hr_data[np.isfinite(hr_data)]
        if len(valid_hr) == 0:
            return "bpm"  # Default

        mean_hr = np.mean(valid_hr)
        return "hz" if mean_hr < 10 else "bpm"

    def _get_effective_threshold(self, hr_data: np.ndarray) -> float:
        """Get threshold in the same unit as the data.

        Args:
            hr_data: Heart rate values.

        Returns:
            Threshold value in data units.
        """
        if self.hr_unit == "bpm":
            return self.hr_threshold_bpm
        elif self.hr_unit == "hz":
            return self.hr_threshold_bpm * self.BPM_TO_HZ
        else:  # auto
            detected_unit = self._detect_hr_unit(hr_data)
            if detected_unit == "hz":
                return self.hr_threshold_bpm * self.BPM_TO_HZ
            return self.hr_threshold_bpm

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate intensity failure mask.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator (not used but required by protocol).

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape
        artificial_mask = np.zeros_like(original_mask)

        # Get HR data
        hr_data = data[self.hr_channel]
        hr_mask = original_mask[self.hr_channel]

        # Get effective threshold
        threshold = self._get_effective_threshold(hr_data)

        # Find minutes where HR exceeds threshold AND is valid
        high_intensity = (hr_data > threshold) & (hr_mask == 1) & np.isfinite(hr_data)

        if not high_intensity.any():
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        # Find contiguous runs of high-intensity minutes
        hi_indices = np.where(high_intensity)[0]
        splits = np.where(np.diff(hi_indices) > 1)[0] + 1
        runs = np.split(hi_indices, splits)

        # Keep only qualifying episodes (>= 5 consecutive points, i.e. > 2 minutes)
        qualifying = [run for run in runs if len(run) >= 5]
        if not qualifying:
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        qualifying_minutes = np.concatenate(qualifying)

        # Mask only qualifying episode minutes
        for t in qualifying_minutes:
            for ch in self.mask_channels:
                if ch < n_channels and original_mask[ch, t] == 1:
                    artificial_mask[ch, t] = 1

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
