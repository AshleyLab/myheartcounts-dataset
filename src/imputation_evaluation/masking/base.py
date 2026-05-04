"""Base classes and protocols for mask generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class MaskResult:
    """Result from a mask generator.

    Attributes:
        artificial_mask: Binary mask of shape (C, T) where 1 indicates positions
            that are artificially masked (will be imputed). Can only be 1 where
            original_mask is also 1.
        applicable: Whether this mask scenario applies to the sample (e.g., a
            sleep mask is only applicable if sleep data exists).
    """

    artificial_mask: np.ndarray
    applicable: bool


class MaskGenerator(Protocol):
    """Protocol for mask generators.

    All mask generators must implement this interface. The invariant is that
    artificial_mask can only mask positions where original_mask is 1 (valid data).
    """

    @property
    def name(self) -> str:
        """Return the name of this mask generator."""
        ...

    @property
    def is_structural(self) -> bool:
        """Return True if this generator only uses original_mask, not data values.

        Structural generators (random_noise, temporal_slice, signal_slice) only need
        the mask pattern, not actual data values. This enables optimizations like
        skipping data loading for these generators.

        Defaults to False (data-dependent) for safety.
        """
        ...

    def generate(
        self,
        data: np.ndarray,
        original_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> MaskResult:
        """Generate an artificial mask for the given sample.

        Args:
            data: Sample data of shape (C, T), may contain NaN.
            original_mask: Binary mask of shape (C, T), 1=valid, 0=missing.
            rng: Random number generator for reproducibility.

        Returns:
            MaskResult with the artificial mask and applicability flag.
        """
        ...
