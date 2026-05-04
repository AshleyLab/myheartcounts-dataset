"""Random noise mask generator.

Masks random patches of contiguous minutes across channels, simulating
brief sensor noise or Bluetooth connection drops.
"""

from __future__ import annotations

import numpy as np

from .base import MaskResult


class RandomNoiseMask:
    """Random patch masking across channels.

    Selects random (channel, start_minute) positions and masks `patch_size`
    contiguous minutes per patch. Creates enough patches to mask approximately
    `mask_ratio` of valid positions.

    Attributes:
        patch_size: Number of consecutive minutes per patch.
        mask_ratio: Target fraction of valid data to mask.
    """

    def __init__(self, patch_size: int = 10, mask_ratio: float = 0.8):
        """Initialize the random noise mask generator.

        Args:
            patch_size: Consecutive minutes per patch.
            mask_ratio: Fraction of valid data to mask.
        """
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    @property
    def name(self) -> str:
        """Return generator name."""
        return "random_noise"

    @property
    def is_structural(self) -> bool:
        """Return True - random noise only depends on mask structure, not data values."""
        return True

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate random patch mask.

        Randomly places contiguous patches of `patch_size` minutes on individual
        channels until approximately `mask_ratio` of valid positions are masked.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator.

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape
        n_patches_per_channel = n_timesteps // self.patch_size

        # Count total valid positions
        total_valid = int(original_mask.sum())
        if total_valid == 0:
            return MaskResult(artificial_mask=np.zeros_like(original_mask), applicable=False)

        # Target number of positions to mask
        target_masked = int(total_valid * self.mask_ratio)

        # Build candidate patches: each patch is (channel, start_minute)
        # Only consider patches that overlap with at least one valid position
        total_candidates = n_channels * n_patches_per_channel
        candidate_channels = np.repeat(np.arange(n_channels), n_patches_per_channel)
        candidate_starts = np.tile(np.arange(n_patches_per_channel) * self.patch_size, n_channels)

        # Shuffle candidates and greedily place patches until target is reached
        order = rng.permutation(total_candidates)
        artificial_mask = np.zeros_like(original_mask)
        masked_count = 0

        for idx in order:
            if masked_count >= target_masked:
                break
            c = candidate_channels[idx]
            t_start = candidate_starts[idx]
            t_end = t_start + self.patch_size
            # Only mask positions that are valid and not already masked
            patch_slice = original_mask[c, t_start:t_end] - artificial_mask[c, t_start:t_end]
            new_masked = int(patch_slice.sum())
            if new_masked > 0:
                artificial_mask[c, t_start:t_end] = original_mask[c, t_start:t_end]
                masked_count += new_masked

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
