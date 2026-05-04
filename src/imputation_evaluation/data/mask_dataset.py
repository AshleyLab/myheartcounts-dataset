"""PyTorch Dataset for parallel mask generation.

Enables using DataLoader with multiple workers to parallelize mask generation
across CPU cores, significantly speeding up imputation evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import datasets as hf_ds

    from data.transforms.nan_transforms import ZeroToNaNTransform
    from imputation_evaluation.masking.base import MaskGenerator


class MaskGenerationDataset(Dataset):
    """PyTorch Dataset that generates masks in worker processes.

    Each worker loads samples from the HuggingFace dataset and applies the mask
    generator independently. Uses deterministic per-sample RNG for reproducibility
    regardless of the number of workers.

    Attributes:
        dataset: The underlying HuggingFace dataset.
        indices: Indices into the dataset for this split.
        generator: Mask generator to apply.
        base_seed: Base seed for deterministic per-sample RNG.
        zero_to_nan_transform: Optional preprocessing transform.
    """

    def __init__(
        self,
        hf_dataset: hf_ds.Dataset,
        indices: list[int],
        generator: MaskGenerator,
        base_seed: int,
        zero_to_nan_transform: ZeroToNaNTransform | None = None,
    ):
        """Initialize the mask generation dataset.

        Args:
            hf_dataset: The HuggingFace dataset to load samples from.
            indices: Indices into the dataset for this split.
            generator: Mask generator to apply to each sample.
            base_seed: Base seed for RNG. Each sample gets seed = base_seed + idx.
            zero_to_nan_transform: Optional transform to apply to values.
        """
        self.dataset = hf_dataset
        self.indices = indices
        self.generator = generator
        self.base_seed = base_seed
        self.zero_to_nan_transform = zero_to_nan_transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """Generate mask for a single sample.

        Args:
            idx: Index into this dataset (0 to len-1).

        Returns:
            Dictionary with:
                - idx: The sample index
                - applicable: Whether the mask is applicable
                - mask: The artificial mask array (C, T), zeros if not applicable
        """
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]

        # Load and preprocess values
        values = torch.as_tensor(sample["values"]).float()
        if self.zero_to_nan_transform is not None:
            values = self.zero_to_nan_transform(values)

        data = values.numpy()
        original_mask = (~np.isnan(data)).astype(np.float32)

        # Deterministic RNG per sample (reproducible across runs and worker counts)
        rng = np.random.default_rng(self.base_seed + idx)

        # Generate mask
        result = self.generator.generate(data, original_mask, rng)

        # Return empty mask if not applicable
        if result.applicable:
            mask = result.artificial_mask
        else:
            mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        return {
            "idx": idx,
            "applicable": result.applicable,
            "mask": mask,
        }
