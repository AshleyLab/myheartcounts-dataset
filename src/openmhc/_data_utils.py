"""Public utilities for accessing the official train/val/test splits.

These let custom imputers fit themselves and look up per-sample
metadata without depending on the internal evaluation harness.

    >>> import openmhc
    >>> for data, mask in openmhc.iter_train_data():
    ...     # data: (B, 19, 1440) float32, NaN at missing positions
    ...     # mask: (B, 19, 1440) float32, 1 = observed, 0 = missing
    ...     pass
    >>> meta = openmhc.load_sample_metadata("val")
    >>> meta[0]
    {'sample_idx': 0, 'user_id': '...', 'date': '2024-...'}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np

from openmhc._dataset import Version
from openmhc._evaluate import _DatasetPaths

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_VALID_SPLITS = ("train", "val", "test")


def _make_data_config(
    data_dir: str | Path | None,
    version: Version,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    """Build the imputation DataConfig used by both utilities and the harness."""
    from imputation_evaluation.config import DataConfig

    paths = _DatasetPaths.resolve(data_dir, version=version)
    paths.require("daily_hf", "splits_file")
    return DataConfig(
        daily_hf_dir=str(paths.daily_hf),
        split_file=str(paths.splits_file),
        split_seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        num_eval_workers=1,
    )


def iter_split_data(
    split: Literal["train", "val", "test"],
    version: Version,
    data_dir: str | Path | None = None,
    batch_size: int = 5000,
    num_workers: int = 0,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(data, mask)`` numpy batches from one official split.

    Each batch is shape ``(B, 19, 1440)``, dtype ``float32``. ``data``
    has ``NaN`` at naturally missing positions; ``mask`` is ``1.0`` where
    observed and ``0.0`` where missing.

    Streams via the same DataLoader the evaluation harness uses, so QA
    filters and preprocessing match. Order is deterministic and aligns
    with :func:`load_sample_metadata` (sample-local index ``k`` is the
    ``k``-th yielded sample across batches).

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        version: ``"xs"`` or ``"full"``. Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        data_dir: Override for the dataset root.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        seed: Random seed (only used if no split file is provided).

    Yields:
        Tuples ``(data, mask)``, each of shape ``(B, 19, 1440)`` and
        dtype ``float32``.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. Valid splits: {_VALID_SPLITS}"
        )
    from imputation_evaluation.data.data_loader import ImputationDataLoader

    cfg = _make_data_config(data_dir, version, batch_size, num_workers, seed)
    loaded = ImputationDataLoader(cfg).load_splits(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    loader = {
        "train": loaded.train_loader,
        "val": loaded.val_loader,
        "test": loaded.test_loader,
    }[split]
    for batch in loader:
        data = batch[0] if not isinstance(batch, dict) else batch["values"]
        mask = batch[1] if not isinstance(batch, dict) else batch["mask"]
        yield (
            np.asarray(data, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
        )


def iter_train_data(
    version: Version,
    data_dir: str | Path | None = None,
    batch_size: int = 5000,
    num_workers: int = 0,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(data, mask)`` numpy batches from the official train split.

    Convenience wrapper around :func:`iter_split_data` with
    ``split="train"``. See that function for details.
    """
    return iter_split_data(
        "train",
        version=version,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )


def load_sample_metadata(
    split: Literal["train", "val", "test"],
    version: Version,
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> list[dict]:
    """Return per-sample metadata for one split.

    Lightweight: only reads ``user_id`` and ``date`` columns from the HF
    dataset, does not materialize tensor values. Uses the same QA
    filters and user-level splits as the evaluation harness, so
    ``sample_idx`` aligns with the position in
    :func:`iter_train_data` (for ``train``) or in the eval data loaders
    (for ``val`` / ``test``).

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        version: ``"xs"`` or ``"full"``. Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        data_dir: Override for the dataset root.
        seed: Random seed (only used if no split file is provided).

    Returns:
        A list of dicts, one per sample, with keys ``sample_idx`` (int,
        split-local position), ``user_id`` (str), ``date`` (str in
        ``YYYY-MM-DD`` form).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}. Valid splits: {_VALID_SPLITS}"
        )

    from imputation_evaluation.data.data_loader import ImputationDataLoader

    cfg = _make_data_config(data_dir, version, batch_size=5000, num_workers=0, seed=seed)
    split_indices, all_user_ids, all_dates = ImputationDataLoader(cfg).load_split_indices()

    indices = split_indices[split]
    return [
        {
            "sample_idx": split_local_idx,
            "user_id": all_user_ids[hf_idx],
            "date": all_dates[hf_idx],
        }
        for split_local_idx, hf_idx in enumerate(indices)
    ]
