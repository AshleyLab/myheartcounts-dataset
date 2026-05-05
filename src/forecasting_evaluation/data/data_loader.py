"""Data loading utilities for downstream evaluation.

Reuses label attachment and user split logic from baseline_datamodule.py
for consistency with the Lightning-based pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import datasets as hf_ds
import datasets.features.features as hf_features

from downstream_evaluation.data.splits import load_split_file, random_split_users

if TYPE_CHECKING:
    from forecasting_evaluation.config import DataConfig

logger = logging.getLogger(__name__)

SAMPLE_INDEX_REQUIRED_MSG = (
    "Forecasting evaluation requires data.sample_index_file. "
    "Please generate it first with scripts/precompute_forecasting_inputs.py."
)


def _ensure_legacy_hf_list_feature_compat() -> None:
    """Allow old saved datasets with ``"_type": "List"`` to load on newer datasets."""
    list_type = getattr(hf_features, "List", None)
    sequence_type = getattr(hf_features, "Sequence", None)
    if sequence_type is None or list_type is sequence_type:
        return
    if is_dataclass(sequence_type) and not is_dataclass(list_type):
        hf_features.List = sequence_type
        logger.warning(
            "Patched datasets.features.features.List to Sequence for legacy HF dataset metadata"
        )


class ForecastingDataLoader:
    """Unified data loading with label attachment and user splits.

    Reuses patterns from baseline_datamodule.py to ensure consistency
    with the Lightning-based pipeline.
    """

    def __init__(self, config: DataConfig):
        """Initialize data loader.

        Args:
            config: Data configuration with paths and split parameters.
        """
        self.config = config
        self.trajectory_hf_dir = Path(config.trajectory_hf_dir)
        self.task_name = config.task_name

    def load_splits(self) -> tuple[hf_ds.Dataset, hf_ds.Dataset, hf_ds.Dataset]:
        """Load data, attach labels, filter, and split by user.

        Returns:
            Tuple of (train_ds, val_ds, test_ds) HuggingFace datasets.
        """
        sample_index_path = self._require_sample_index_file()

        # 1. Load full dataset
        logger.info(f"Loading trajectory dataset from {self.trajectory_hf_dir}")
        _ensure_legacy_hf_list_feature_compat()
        full_ds = hf_ds.load_from_disk(str(self.trajectory_hf_dir))
        logger.info(f"Loaded {len(full_ds)} trajectory samples")

        if (
            hasattr(self.config, "max_samples")
            and self.config.max_samples
            and len(full_ds) > self.config.max_samples
        ):
            logger.info(f"Subsampling to max_samples={self.config.max_samples}")
            full_ds = full_ds.select(range(self.config.max_samples))


        all_users = sorted({str(user_id) for user_id in full_ds["user_id"]})

        # 2. Filter
        sample_index_data = {}
        with sample_index_path.open("r", encoding="utf-8") as f:
            sample_index_data = json.load(f)

        allowed_user_ids = set(map(str, sample_index_data.keys()))
        all_users = [user_id for user_id in all_users if user_id in allowed_user_ids]


        # 4. Get user splits
        user_splits = self._get_user_splits(all_users)

        # Enforce both split file and sample-index/full-dataset availability.
        eligible_users = set(all_users)
        user_splits = {
            split_name: {str(user_id) for user_id in split_users} & eligible_users
            for split_name, split_users in user_splits.items()
        }

        train_sample_count = sum(len(sample_index_data.get(uid, [])) for uid in user_splits["train"])
        val_sample_count = sum(len(sample_index_data.get(uid, [])) for uid in user_splits["validation"])
        test_sample_count = sum(len(sample_index_data.get(uid, [])) for uid in user_splits["test"])

        logger.info(
            "Effective split users after sample_index/full_ds filter: "
            "train=%d, val=%d, test=%d; sample_index samples: train=%d, val=%d, test=%d",
            len(user_splits["train"]),
            len(user_splits["validation"]),
            len(user_splits["test"]),
            train_sample_count,
            val_sample_count,
            test_sample_count,
        )

        # 5. Filter by split
        # Passing num_proc=1 still enables the datasets multiprocessing path.
        # Keep single-worker runs truly serial so restricted test runners do not
        # need to create a multiprocessing manager socket.
        filter_num_proc = self.config.num_workers if self.config.num_workers > 1 else None
        train_ds = full_ds.filter(
            lambda ex: ex["user_id"] in user_splits["train"],
            num_proc=filter_num_proc,
        )
        val_ds = full_ds.filter(
            lambda ex: ex["user_id"] in user_splits["validation"],
            num_proc=filter_num_proc,
        )
        test_ds = full_ds.filter(
            lambda ex: ex["user_id"] in user_splits["test"],
            num_proc=filter_num_proc,
        )

        logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} samples")

        return train_ds, val_ds, test_ds

    def _require_sample_index_file(self) -> Path:
        """Return a validated sample-index path required by forecasting evaluation."""
        sample_index_value = self.config.sample_index_file
        if not sample_index_value:
            raise ValueError(SAMPLE_INDEX_REQUIRED_MSG)

        sample_index_path = Path(sample_index_value)
        if not sample_index_path.exists():
            raise FileNotFoundError(
                f"sample_index_file not found: {sample_index_path}. {SAMPLE_INDEX_REQUIRED_MSG}"
            )

        return sample_index_path

    def _get_user_splits(self, user_ids: list[str]) -> dict[str, set[str]]:
        """Load or generate user splits.

        Uses the same functions as baseline_datamodule.py to ensure consistency.
        """
        if self.config.split_file:
            split_path = Path(self.config.split_file)
            if split_path.exists():
                logger.info(f"Loading splits from {split_path}")
                return load_split_file(split_path)

        # Generate random split
        logger.info(
            f"Generating random split for {len(user_ids)} users (seed={self.config.split_seed})"
        )
        return random_split_users(
            user_ids,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.split_seed,
        )
