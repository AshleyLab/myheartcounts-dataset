"""Signal slice mask generator.

Masks entire channels for the day, simulating sensor crashes (Mode A)
or device-not-worn scenarios (Mode B).
"""

from __future__ import annotations

import math

import numpy as np

from .base import MaskResult


class SignalSliceMask:
    """Channel-level masking for entire day.

    Two sub-modes selected 50/50 per sample:
    - Mode A: Randomly select individual channels to drop for entire day.
    - Mode B: Drop all channels of a randomly selected device group.

    Attributes:
        mask_ratio: Fraction of channels to drop in Mode A.
        device_groups: Mapping of device name to channel indices for Mode B.
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        device_groups: dict[str, list[int]] | None = None,
    ):
        """Initialize the signal slice mask generator.

        Args:
            mask_ratio: Fraction of channels to drop in Mode A.
            device_groups: Device name to channel indices mapping.
        """
        self.mask_ratio = mask_ratio
        self.device_groups = device_groups or {
            "iphone": [0, 1, 2],
            "watch": [3, 4, 5, 6],
        }

    @property
    def name(self) -> str:
        """Return generator name."""
        return "signal_slice"

    @property
    def is_structural(self) -> bool:
        """Return True - signal slice only depends on mask structure, not data values."""
        return True

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate signal slice mask.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator.

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape
        artificial_mask = np.zeros_like(original_mask)

        # Find channels that have any valid data
        valid_per_channel = original_mask.sum(axis=1)  # (C,)
        valid_channels = np.where(valid_per_channel > 0)[0]

        if len(valid_channels) == 0:
            return MaskResult(artificial_mask=artificial_mask, applicable=False)

        # Choose mode: 50/50 between Mode A (individual channels) and Mode B (device group)
        use_mode_a = rng.random() < 0.5

        if use_mode_a:
            # Mode A: Drop random individual channels
            n_to_drop = max(1, math.ceil(len(valid_channels) * self.mask_ratio))
            channels_to_drop = rng.choice(valid_channels, size=n_to_drop, replace=False)
        else:
            # Mode B: Drop a random device group
            available_groups = []
            for group_name, group_channels in self.device_groups.items():
                # Check if any channel in this group has valid data
                if any(ch in valid_channels for ch in group_channels):
                    available_groups.append(group_name)

            if not available_groups:
                # Fall back to Mode A
                n_to_drop = max(1, math.ceil(len(valid_channels) * self.mask_ratio))
                channels_to_drop = rng.choice(valid_channels, size=n_to_drop, replace=False)
            else:
                # Select random device group
                group_name = rng.choice(available_groups)
                channels_to_drop = [
                    ch for ch in self.device_groups[group_name] if ch in valid_channels
                ]

        # Mask entire day for selected channels
        for ch in channels_to_drop:
            artificial_mask[ch, :] = original_mask[ch, :]

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
