"""Mask cache generation for imputation evaluation.

Provides parallel mask generation using DataLoader workers and efficient
bit-packed storage for memory efficiency.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import DataLoader

from imputation_evaluation.data.mask_dataset import MaskGenerationDataset

if TYPE_CHECKING:
    import datasets as hf_ds

    from data.transforms.nan_transforms import ZeroToNaNTransform
    from imputation_evaluation.masking.base import MaskGenerator

logger = logging.getLogger(__name__)

# Constants
N_CHANNELS = 19
N_TIMESTEPS = 1440


def pack_masks(masks: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pack binary masks to bits for memory efficiency.

    Args:
        masks: Binary masks of shape (N, C, T) with values 0 or 1.

    Returns:
        Tuple of (packed_array, original_shape) where packed_array is uint8.
    """
    binary = (masks > 0.5).astype(np.uint8)
    flat = binary.reshape(-1)
    packed = np.packbits(flat)
    return packed, masks.shape


def unpack_masks(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Unpack bit-packed masks to float32.

    Args:
        packed: Bit-packed uint8 array.
        shape: Original shape (N, C, T).

    Returns:
        Unpacked masks as float32 array of given shape.
    """
    flat = np.unpackbits(packed)
    n_elements = int(np.prod(shape))
    return flat[:n_elements].reshape(shape).astype(np.float32)


@dataclass
class ScenarioMasks:
    """Pre-generated masks for a single scenario.

    Stores masks in bit-packed format for 32x memory reduction.
    Provides efficient lookup by global sample index.

    Attributes:
        indices: Sorted global indices of applicable samples.
        masks_packed: Bit-packed masks (uint8).
        shape: Original shape (N_applicable, C, T).
    """

    indices: np.ndarray
    masks_packed: np.ndarray
    shape: tuple[int, ...]

    # Lazy-initialized lookup structures
    _index_set: set[int] = field(default_factory=set, repr=False)
    _index_to_pos: dict[int, int] = field(default_factory=dict, repr=False)
    _unpacked_cache: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Build lookup structures for fast index access."""
        self._index_set = set(self.indices.tolist())
        self._index_to_pos = {int(idx): pos for pos, idx in enumerate(self.indices)}

    @property
    def n_applicable(self) -> int:
        """Return number of applicable samples."""
        return self.shape[0]

    def contains(self, global_idx: int) -> bool:
        """Check if a global index has a mask in this scenario."""
        return global_idx in self._index_set

    def get_position(self, global_idx: int) -> int | None:
        """Get the position in the masks array for a global index."""
        return self._index_to_pos.get(global_idx)

    def _unpack_positions(self, positions: list[int]) -> np.ndarray:
        """Selectively unpack only the requested positions from bit-packed storage.

        Avoids unpacking the full array, which is critical for memory efficiency
        in parallel evaluation where each worker would otherwise unpack all masks.

        Args:
            positions: List of positions (0 to n_applicable-1).

        Returns:
            Masks of shape (len(positions), C, T) as float32.
        """
        if not positions:
            return np.empty((0, self.shape[1], self.shape[2]), dtype=np.float32)

        elements_per_mask = self.shape[1] * self.shape[2]
        bytes_per_mask = int(np.ceil(elements_per_mask / 8))

        starts = np.array(positions) * bytes_per_mask
        packed_subset = np.stack([self.masks_packed[s : s + bytes_per_mask] for s in starts])

        unpacked = np.unpackbits(packed_subset, axis=1)[:, :elements_per_mask]
        return unpacked.reshape(-1, self.shape[1], self.shape[2]).astype(np.float32)

    def get_masks(self, positions: list[int]) -> np.ndarray:
        """Get masks for given positions in the masks array.

        Args:
            positions: List of positions (0 to n_applicable-1).

        Returns:
            Masks of shape (len(positions), C, T).
        """
        if self._unpacked_cache is not None:
            return self._unpacked_cache[positions]
        return self._unpack_positions(positions)

    def get_mask(self, global_idx: int) -> np.ndarray | None:
        """Get mask for a single global index.

        Args:
            global_idx: Global sample index.

        Returns:
            Mask of shape (C, T) or None if not applicable.
        """
        pos = self.get_position(global_idx)
        if pos is None:
            return None
        if self._unpacked_cache is not None:
            return self._unpacked_cache[pos]
        return self._unpack_positions([pos])[0]

    def save(self, filepath: Path) -> None:
        """Save to compressed npz file.

        Args:
            filepath: Path to save the npz file.
        """
        np.savez_compressed(
            filepath,
            indices=self.indices,
            masks_packed=self.masks_packed,
            shape=np.array(self.shape),
        )

    @classmethod
    def load(cls, filepath: Path) -> ScenarioMasks:
        """Load from npz file.

        Args:
            filepath: Path to the npz file.

        Returns:
            Loaded ScenarioMasks instance.
        """
        data = np.load(filepath)
        return cls(
            indices=data["indices"],
            masks_packed=data["masks_packed"],
            shape=tuple(data["shape"].tolist()),
        )


class MaskCache:
    """In-memory cache of generated masks for all scenarios and splits.

    Provides efficient batch lookup for evaluation.
    """

    def __init__(self):
        """Initialize empty cache."""
        # Structure: masks[split_name][scenario_name] = ScenarioMasks
        self._masks: dict[str, dict[str, ScenarioMasks]] = {}

    def add(self, split: str, scenario: str, masks: ScenarioMasks) -> None:
        """Add masks for a scenario/split combination.

        Args:
            split: Split name (e.g., "val", "test").
            scenario: Scenario name (e.g., "random_noise").
            masks: Pre-generated masks for this combination.
        """
        if split not in self._masks:
            self._masks[split] = {}
        self._masks[split][scenario] = masks

    def get(self, split: str, scenario: str) -> ScenarioMasks | None:
        """Get masks for a scenario/split combination.

        Args:
            split: Split name.
            scenario: Scenario name.

        Returns:
            ScenarioMasks or None if not found.
        """
        return self._masks.get(split, {}).get(scenario)

    def get_scenarios(self, split: str) -> list[str]:
        """Get list of scenario names for a split.

        Args:
            split: Split name.

        Returns:
            List of scenario names.
        """
        return list(self._masks.get(split, {}).keys())

    def get_splits(self) -> list[str]:
        """Get list of split names.

        Returns:
            List of split names.
        """
        return list(self._masks.keys())

    def get_batch_masks(
        self, split: str, scenario: str, batch_global_indices: list[int]
    ) -> tuple[list[int], np.ndarray]:
        """Get masks for applicable samples in a batch.

        Args:
            split: Split name.
            scenario: Scenario name.
            batch_global_indices: List of global indices for samples in the batch.

        Returns:
            Tuple of (local_indices, masks) where:
                - local_indices: Indices within the batch (0 to batch_size-1) that are applicable
                - masks: Array of shape (n_applicable, C, T) with masks for applicable samples
        """
        scenario_masks = self.get(split, scenario)
        if scenario_masks is None:
            return [], np.empty((0, N_CHANNELS, N_TIMESTEPS), dtype=np.float32)

        # Find which batch samples have masks
        local_indices = []
        positions = []
        for local_idx, global_idx in enumerate(batch_global_indices):
            pos = scenario_masks.get_position(global_idx)
            if pos is not None:
                local_indices.append(local_idx)
                positions.append(pos)

        if not positions:
            return [], np.empty((0, N_CHANNELS, N_TIMESTEPS), dtype=np.float32)

        masks = scenario_masks.get_masks(positions)
        return local_indices, masks

    def get_applicable_indices(self, split: str) -> set[int]:
        """Return the union of all applicable sample indices across scenarios for a split.

        Useful for pre-filtering DataLoaders to skip samples that have no masks
        in any scenario, avoiding wasted data loading and inference.

        Args:
            split: Split name (e.g., "val", "test").

        Returns:
            Set of split-local sample indices that have at least one mask.
        """
        applicable: set[int] = set()
        for scenario_masks in self._masks.get(split, {}).values():
            applicable.update(scenario_masks.indices.tolist())
        return applicable

    def get_single_mask(self, split: str, scenario: str, global_idx: int) -> np.ndarray | None:
        """Get mask for a single sample by its split-local index.

        Convenience method for multi-day evaluation, where per-day masks
        are looked up individually and assembled into full-window masks.

        Args:
            split: Split name (e.g., "val", "test").
            scenario: Scenario name (e.g., "random_noise").
            global_idx: Split-local sample index.

        Returns:
            Mask of shape (C, T) or None if not applicable.
        """
        scenario_masks = self.get(split, scenario)
        if scenario_masks is None:
            return None
        return scenario_masks.get_mask(global_idx)

    def save(self, masks_dir: Path) -> None:
        """Save all masks to directory.

        Args:
            masks_dir: Directory to save masks to.
        """
        masks_dir.mkdir(parents=True, exist_ok=True)

        for split in self._masks:
            split_dir = masks_dir / split
            split_dir.mkdir(exist_ok=True)

            for scenario, scenario_masks in self._masks[split].items():
                filepath = split_dir / f"{scenario}.npz"
                scenario_masks.save(filepath)
                logger.info(
                    f"Saved {scenario_masks.n_applicable} masks "
                    f"for {split}/{scenario} to {filepath}"
                )

    @classmethod
    def load(
        cls,
        masks_dir: Path,
        scenarios: list[str],
        splits: list[str],
    ) -> MaskCache:
        """Load masks from directory.

        Args:
            masks_dir: Directory containing saved masks.
            scenarios: List of scenario names to load.
            splits: List of split names to load.

        Returns:
            Loaded MaskCache instance.
        """
        cache = cls()

        for split in splits:
            split_dir = masks_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            for scenario in scenarios:
                filepath = split_dir / f"{scenario}.npz"
                if not filepath.exists():
                    logger.warning(f"Mask file not found: {filepath}")
                    continue

                scenario_masks = ScenarioMasks.load(filepath)
                cache.add(split, scenario, scenario_masks)
                logger.info(
                    f"Loaded {scenario_masks.n_applicable} masks "
                    f"for {split}/{scenario} from {filepath}"
                )

        return cache


class MaskCacheGenerator:
    """Generates masks for all scenarios using parallel DataLoader workers.

    Uses MaskGenerationDataset to parallelize mask generation across CPU cores.
    """

    def __init__(
        self,
        hf_dataset: hf_ds.Dataset,
        zero_to_nan_transform: ZeroToNaNTransform | None = None,
        num_workers: int = 4,
        batch_size: int = 1000,
    ):
        """Initialize the mask cache generator.

        Args:
            hf_dataset: The HuggingFace dataset to generate masks for.
            zero_to_nan_transform: Optional preprocessing transform.
            num_workers: Number of worker processes for parallel generation.
            batch_size: Batch size for DataLoader.
        """
        self.hf_dataset = hf_dataset
        self.zero_to_nan_transform = zero_to_nan_transform
        self.num_workers = num_workers
        self.batch_size = batch_size

    def generate(
        self,
        split_indices: dict[str, list[int]],
        generators: list[MaskGenerator],
        base_seed: int,
    ) -> MaskCache:
        """Generate masks for all scenarios and splits.

        Args:
            split_indices: Dict mapping split name to list of global indices.
            generators: List of mask generators (one per scenario).
            base_seed: Base random seed for reproducibility.

        Returns:
            MaskCache with pre-generated masks for all combinations.
        """
        cache = MaskCache()

        for split_name, indices in split_indices.items():
            if not indices:
                logger.info(f"Skipping empty split: {split_name}")
                continue

            logger.info(f"Generating masks for {split_name} split ({len(indices)} samples)...")

            # Compute seed offset for this split (stable across runs, unlike hash())
            split_seed_offset = int(hashlib.md5(split_name.encode()).hexdigest(), 16) % 1000
            split_base_seed = base_seed + split_seed_offset

            for generator in generators:
                scenario_name = generator.name

                # Compute scenario seed (stable across runs, unlike hash())
                scenario_seed = (
                    split_base_seed
                    + int(hashlib.md5(scenario_name.encode()).hexdigest(), 16) % 10000
                )

                scenario_masks = self._generate_scenario(
                    indices=indices,
                    generator=generator,
                    base_seed=scenario_seed,
                )

                cache.add(split_name, scenario_name, scenario_masks)
                logger.info(
                    f"  {scenario_name}: {scenario_masks.n_applicable}/{len(indices)} applicable"
                )

        return cache

    def _generate_scenario(
        self,
        indices: list[int],
        generator: MaskGenerator,
        base_seed: int,
    ) -> ScenarioMasks:
        """Generate masks for a single scenario.

        Args:
            indices: Global indices for this split.
            generator: Mask generator for this scenario.
            base_seed: Base seed for this scenario/split.

        Returns:
            ScenarioMasks with pre-generated masks.
        """
        # Create dataset for parallel generation
        dataset = MaskGenerationDataset(
            hf_dataset=self.hf_dataset,
            indices=indices,
            generator=generator,
            base_seed=base_seed,
            zero_to_nan_transform=self.zero_to_nan_transform,
        )

        # Create DataLoader
        persistent = self.num_workers > 0
        prefetch = 2 if self.num_workers > 0 else None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

        # Collect masks from all batches, packing to bits incrementally
        # to avoid accumulating large float32 arrays in memory.
        # Store split-local indices (0 to len-1), NOT HF dataset indices
        # This matches how the evaluator's DataLoader yields batches
        applicable_indices = []
        packed_chunks = []
        elements_per_mask = N_CHANNELS * N_TIMESTEPS
        elements_per_mask // 8  # exact since 19*1440=27360 is divisible by 8

        for batch in loader:
            batch_applicable = batch["applicable"].numpy()
            batch_masks = batch["mask"].numpy()
            batch_local_indices = batch["idx"].numpy()

            # Filter to applicable samples
            applicable_mask = batch_applicable.astype(bool)
            if not applicable_mask.any():
                continue

            applicable_batch_masks = batch_masks[applicable_mask]
            applicable_batch_indices = batch_local_indices[applicable_mask]

            applicable_indices.extend(applicable_batch_indices.tolist())

            # Pack this batch's masks to bits immediately (huge memory savings)
            binary = (applicable_batch_masks > 0.5).astype(np.uint8)
            flat = binary.reshape(len(binary), -1)
            packed_batch = np.packbits(flat, axis=1)
            packed_chunks.append(packed_batch.reshape(-1))

        # Concatenate packed chunks
        if packed_chunks:
            masks_packed = np.concatenate(packed_chunks)
            n_applicable = len(applicable_indices)
            shape = (n_applicable, N_CHANNELS, N_TIMESTEPS)
        else:
            masks_packed = np.array([], dtype=np.uint8)
            shape = (0, N_CHANNELS, N_TIMESTEPS)

        indices_array = np.array(applicable_indices, dtype=np.int64)

        return ScenarioMasks(
            indices=indices_array,
            masks_packed=masks_packed,
            shape=shape,
        )
