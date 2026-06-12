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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

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


# ---------------------------------------------------------------------------
# Per-user lazy-read context for personalized imputers
# ---------------------------------------------------------------------------


@dataclass
class EvalUserContext:
    """Per-user lazy-read handle for personalized imputers.

    Holds the memory-mapped HF dataset reference plus the per-user row
    index built once at imputer ``__init__``. Designed for the
    eval-time lazy state pattern: each ``impute()`` call streams the
    current user's samples on demand via :meth:`iter_user_samples`
    instead of holding all val+test samples' contributions eagerly.

    Memory-mapped HF datasets are fork-safe (read-only Arrow files), so
    forked workers inherit the dataset reference cheaply. The per-user
    index dicts are also fork-safe — small Python objects copied
    on-write but rarely written after init.

    Attributes:
        hf_dataset: The filtered HuggingFace dataset (memory-mapped).
            Typed as ``Any`` to avoid eagerly importing ``datasets`` at
            module load time.
        user_to_hf_indices: ``{user_id: np.ndarray[int]}`` mapping each
            val/test user to their HF row indices, sorted ascending.
        user_to_split: ``{user_id: "val" | "test"}`` telling which split
            a known user came from (mutually exclusive under the
            canonical user-level split file).
        split_hf_indices: ``{split: np.ndarray[int]}`` for val and test
            — the split-wide HF row indices in canonical order. Used to
            recover the split-local ``sample_idx`` for each row when
            keying per-sample contributions for LOSO.
        transform: The ``ZeroToNaNTransform`` instance used by the eval
            data loader (``None`` if ``zero_to_nan`` is disabled).
            Applied lazily inside :meth:`iter_user_samples`.
    """

    hf_dataset: Any
    user_to_hf_indices: dict[str, np.ndarray]
    user_to_split: dict[str, str]
    split_hf_indices: dict[str, np.ndarray]
    transform: Any

    def iter_user_samples(
        self, user_id: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(data, mask)`` per sample for one user, in dataset order.

        Each yielded tuple is shape ``((19, 1440), (19, 1440))`` float32 —
        matches what :func:`iter_split_data` yields per batch item. NaN
        at naturally missing positions in ``data``; ``mask`` is ``1.0``
        where observed and ``0.0`` where missing.

        If ``user_id`` is unknown (not in val/test), yields nothing.
        """
        hf_indices = self.user_to_hf_indices.get(user_id)
        if hf_indices is None:
            return
        transform = self.transform
        for hf_idx in hf_indices:
            row = self.hf_dataset[int(hf_idx)]
            if transform is not None:
                # Transform expects a torch tensor.
                import torch

                values = transform(
                    torch.as_tensor(row["values"], dtype=torch.float32)
                ).numpy()
            else:
                values = np.asarray(row["values"], dtype=np.float32)
            mask = np.isfinite(values).astype(np.float32)
            yield values, mask


def open_eval_user_context(
    version: Version,
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> EvalUserContext:
    """Open the HF dataset for one-shot per-user lazy reads.

    Used by personalized imputers' ``__init__`` to set up the eval-time
    per-user reading context. Performs the same QA-filter + user-level
    split as the harness, then builds a per-user reverse index so that
    ``EvalUserContext.iter_user_samples(user_id)`` can stream just one
    user's rows.

    The returned ``EvalUserContext`` holds an HF Arrow dataset reference
    — memory-mapped, fork-safe, ~zero copy across forked workers.
    """
    from imputation_evaluation.data.data_loader import ImputationDataLoader

    cfg = _make_data_config(
        data_dir, version, batch_size=5000, num_workers=0, seed=seed
    )
    loader = ImputationDataLoader(cfg)
    # Reuse the existing public helper; it loads + filters the dataset
    # once and surfaces the split indices and per-row metadata we need.
    split_indices, all_user_ids, _all_dates = loader.load_split_indices()

    # Re-open + re-filter to obtain a dataset handle. ``load_split_indices``
    # discards it; the re-open uses memory-mapped IO so the second open is
    # nearly free (no second filter pass either, since QA filters are
    # deterministic and idempotent).
    import datasets as hf_ds

    from imputation_evaluation.data.filters import apply_filters

    ds = hf_ds.load_from_disk(cfg.daily_hf_dir)
    filters = loader._build_filters()
    if filters:
        ds = apply_filters(ds, filters)

    user_to_hf: dict[str, list[int]] = {}
    user_to_split: dict[str, str] = {}
    for split_name in ("val", "test"):
        for hf_idx in split_indices[split_name]:
            uid = all_user_ids[hf_idx]
            user_to_hf.setdefault(uid, []).append(int(hf_idx))
            # The canonical user-level split file makes users mutually
            # exclusive across val/test; if the invariant ever breaks we'd
            # rather see the second split win (latest wins on the
            # reassignment) — harmless because contributions stack.
            user_to_split[uid] = split_name

    user_to_hf_arr = {
        u: np.asarray(sorted(idxs), dtype=np.int64) for u, idxs in user_to_hf.items()
    }
    split_hf_indices_arr = {
        split_name: np.asarray(split_indices[split_name], dtype=np.int64)
        for split_name in ("val", "test")
    }

    return EvalUserContext(
        hf_dataset=ds,
        user_to_hf_indices=user_to_hf_arr,
        user_to_split=user_to_split,
        split_hf_indices=split_hf_indices_arr,
        transform=loader._zero_to_nan_transform,
    )
