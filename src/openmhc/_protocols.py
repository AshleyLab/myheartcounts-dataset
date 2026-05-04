"""Duck-typed protocols for MHC-Benchmark model interfaces.

Users implement these interfaces to plug custom models into the benchmark.
No base class inheritance required -- just implement the right methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """Protocol for downstream health prediction encoders.

    Encode weekly sensor tensors into fixed-size embeddings.

    Example:
        >>> class MyEncoder:
        ...     def encode(self, weekly_tensors: np.ndarray) -> np.ndarray:
        ...         # weekly_tensors: (B, 168, 38)
        ...         # Returns: (B, D) embeddings
        ...         return weekly_tensors[:, :, :19].mean(axis=1)
    """

    def encode(self, weekly_tensors: np.ndarray) -> np.ndarray:
        """Map weekly sensor data to fixed-size embeddings.

        Args:
            weekly_tensors: Array of shape (B, 168, 38).
                Columns 0-18 are z-scored sensor values (19 channels, hourly).
                Columns 19-37 are missingness masks (1 = missing, 0 = observed).

        Returns:
            Array of shape (B, D) where D is any embedding dimensionality.
            Must be float32.
        """
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for imputation evaluation.

    Example:
        >>> class MeanImputer:
        ...     def fit(self, data, masks):
        ...         self.means = np.nanmean(data, axis=(0, 2))
        ...     def impute(self, data, observed_mask, target_mask):
        ...         result = data.copy()
        ...         for ch in range(19):
        ...             target = target_mask[:, ch, :] == 1
        ...             result[:, ch, :][target] = self.means[ch]
        ...         return result
    """

    def fit(self, data: np.ndarray, masks: np.ndarray) -> None:
        """Fit on training data.

        Args:
            data: Daily sensor values of shape (N, 19, 1440). NaN at missing
                positions.
            masks: Binary masks of shape (N, 19, 1440). 1 = observed,
                0 = missing.
        """
        ...

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Impute artificially masked positions.

        Args:
            data: Sensor values of shape (N, 19, 1440) with NaN at masked
                positions.
            observed_mask: Binary mask of shape (N, 19, 1440). 1 = originally
                observed, 0 = naturally missing.
            target_mask: Binary mask of shape (N, 19, 1440). 1 = positions to
                impute (a subset of observed_mask).

        Returns:
            Array of shape (N, 19, 1440) with imputed values at target
            positions. Must be float32.
        """
        ...
