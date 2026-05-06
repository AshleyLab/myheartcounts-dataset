"""Base protocol for feature extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import datasets as hf_ds


class FeatureExtractor(Protocol):
    """Protocol for feature extraction from HuggingFace datasets."""

    def extract(self, hf_dataset: hf_ds.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from dataset.

        Args:
            hf_dataset: HuggingFace dataset with 'x', 'user_id', and task_name columns.
            task_name: Name of the label column.

        Returns:
            Tuple of (features, labels, user_ids) where:
            - features: (N, D) float32 array
            - labels: (N,) int64 array
            - user_ids: (N,) object array of user ID strings
        """
        ...
