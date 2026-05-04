"""Temporal slice mask generator.

Masks contiguous time blocks across ALL channels, simulating device
downtime (e.g., showering, charging, battery depletion).
"""

from __future__ import annotations

import math

import numpy as np

from .base import MaskResult


class TemporalSliceMask:
    """Contiguous temporal block masking across all channels.

    Selects contiguous time windows of random length and masks ALL channels
    for each window. Creates enough blocks to mask approximately `mask_ratio`
    of valid time steps.

    Attributes:
        mask_ratio: Target fraction of valid timesteps to mask.
        min_block_size: Minimum contiguous block size in minutes.
        max_block_size: Maximum contiguous block size in minutes.
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        min_block_size: int = 30,
        max_block_size: int = 60,
    ):
        """Initialize the temporal slice mask generator.

        Args:
            mask_ratio: Fraction of valid timesteps to mask.
            min_block_size: Min block size in minutes.
            max_block_size: Max block size in minutes.
        """
        self.mask_ratio = mask_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    @property
    def name(self) -> str:
        """Return generator name."""
        return "temporal_slice"

    @property
    def is_structural(self) -> bool:
        """Return True - temporal slice only depends on mask structure, not data values."""
        return True

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate temporal slice mask.

        Args:
            data: Sample data of shape (C, T).
            original_mask: Binary mask of shape (C, T), 1=valid.
            rng: Random number generator.

        Returns:
            MaskResult with artificial mask.
        """
        n_channels, n_timesteps = data.shape

        # Find timesteps that have at least one valid channel
        valid_per_timestep = original_mask.sum(axis=0)  # (T,)
        n_valid_timesteps = int((valid_per_timestep > 0).sum())

        if n_valid_timesteps == 0:
            return MaskResult(artificial_mask=np.zeros_like(original_mask), applicable=False)

        # Target number of timesteps to mask
        int(n_valid_timesteps * self.mask_ratio)

        # Estimate number of blocks needed using the random-interval coverage
        # formula: coverage ≈ 1 - (1 - s/L)^n, solved for n.
        avg_block_size = (self.min_block_size + self.max_block_size) / 2
        coverage_per_block = avg_block_size / n_timesteps
        n_blocks = max(
            1,
            math.ceil(math.log(1 - self.mask_ratio) / math.log(1 - coverage_per_block)),
        )

        # Generate all block starts and sizes at once (vectorized)
        block_sizes = rng.integers(self.min_block_size, self.max_block_size + 1, size=n_blocks)
        max_starts = np.maximum(1, n_timesteps - block_sizes)
        block_starts = rng.integers(0, max_starts)
        block_ends = np.minimum(block_starts + block_sizes, n_timesteps)

        # Create timestep mask using vectorized range expansion
        timestep_mask = np.zeros(n_timesteps, dtype=np.float32)
        for start, end in zip(block_starts, block_ends):
            timestep_mask[start:end] = 1

        # Only mask timesteps that have valid data
        timestep_mask = timestep_mask * (valid_per_timestep > 0).astype(np.float32)

        # Mask all valid positions at selected timesteps (vectorized broadcast)
        artificial_mask = original_mask * timestep_mask[np.newaxis, :]

        return MaskResult(artificial_mask=artificial_mask, applicable=True)
