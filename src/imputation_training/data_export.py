"""HDF5 export utility for PyPOTS training.

PyPOTS' ``model.fit(train_set=..., file_type='hdf5')`` consumes
``.h5`` files containing a dataset ``"X"`` of shape ``(N, T, C)`` with
NaN marking missing values. For val/test splits PyPOTS additionally
wants an ``"X_ori"`` with the (un-artificially-masked) ground truth so
it can compute a per-epoch validation metric.

This module streams batches from
:class:`imputation_evaluation.data.data_loader.ImputationDataLoader` —
the same loader the eval pipeline uses, so train/val/test splits and QA
filtering are guaranteed identical — and writes the resulting tensors
to disk. The output directory is content-addressed (8-char SHA-256
hash of the relevant config fields) so different split/preprocessing
combinations never collide in the same cache.

Ported and simplified from ``MHC-benchmark/src/pypots_training/data_export.py``.
The private-repo dependency on ``data.normalization.ChannelStats`` is
replaced by :class:`imputation_training.normalization.ChannelStats`,
which loads from the same ``normalization_stats.json`` schema OpenMHC
ships in its dataset cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from imputation_evaluation.config import DataConfig
from imputation_evaluation.data.data_loader import ImputationDataLoader
from imputation_training.normalization import (
    ChannelStats,
    copy_stats_file,
    derive_stats_path_from_daily_hf,
)

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def h5_cache_subdir(base_dir: str | Path, data_config, h5_export_config) -> Path:
    """Return a content-addressed subdirectory for H5 files.

    The hash key is every config field that affects H5 content
    (splits, sample limits, filters, normalization, val masking).
    Different combinations therefore never share the same cache dir,
    eliminating stale-cache collisions.
    """
    key = {
        "daily_hf_dir": str(data_config.daily_hf_dir),
        "split_file": str(data_config.split_file),
        "train_ratio": data_config.train_ratio,
        "val_ratio": data_config.val_ratio,
        "split_seed": data_config.split_seed,
        "max_samples_per_split": data_config.max_samples_per_split,
        "n_days": data_config.n_days,
        "preprocessing": asdict(data_config.preprocessing),
        "filters": asdict(data_config.filters),
        "normalize": h5_export_config.normalize,
        "val_mask_ratio": h5_export_config.val_mask_ratio,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return Path(base_dir) / digest


def h5_files_exist(output_dir: str | Path, splits: tuple[str, ...] = SPLITS) -> bool:
    """Return whether an ``<split>.h5`` file exists for every split.

    Args:
        output_dir: Directory expected to hold the per-split H5 files.
        splits: Split names to check for.

    Returns:
        ``True`` only if every split's H5 file is present.
    """
    output_dir = Path(output_dir)
    return all((output_dir / f"{split}.h5").exists() for split in splits)


def export_splits_to_h5(
    data_config: DataConfig,
    output_dir: str | Path,
    *,
    splits: tuple[str, ...] = SPLITS,
    chunk_size: int = 1000,
    overwrite: bool = False,
    val_mask_ratio: float = 0.2,
    normalize: bool = True,
    normalization_stats_path: str | Path | None = None,
) -> dict[str, Path]:
    """Export train/val/test splits to H5 for PyPOTS.

    Reuses :class:`ImputationDataLoader` so QA filters, preprocessing,
    and user-level splits are byte-identical to the eval pipeline.

    For val/test splits we additionally write ``X_ori`` (the
    un-artificially-masked ground truth) and apply a per-sample random
    patch mask via :class:`imputation_evaluation.masking.random_noise.RandomNoiseMask`
    so PyPOTS' own validation metric has held-out positions to score
    against.

    Args:
        data_config: Data settings, same dataclass the eval CLI uses.
        output_dir: Where to write the H5 files. Created if missing.
        splits: Which splits to export.
        chunk_size: H5 chunk size (rows).
        overwrite: If False, skip the export when all H5 files already exist.
        val_mask_ratio: Fraction of observed positions to mask in val/test.
        normalize: Z-score the continuous channels before writing.
        normalization_stats_path: Path to ``normalization_stats.json``.
            If None and ``normalize`` is True, falls back to
            ``<daily_hf_dir>/../normalization_stats.json`` (the OpenMHC
            dataset-cache layout). The chosen stats file is also copied
            into ``output_dir`` for the inference side to consume.

    Returns:
        Dict mapping split name to the H5 file path.
    """
    from imputation_evaluation.masking.random_noise import RandomNoiseMask

    output_dir = Path(output_dir)
    if not overwrite and h5_files_exist(output_dir, splits):
        logger.info("H5 files already exist in %s; skipping export.", output_dir)
        return {split: output_dir / f"{split}.h5" for split in splits}

    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ImputationDataLoader(data_config)
    loaded = loader.load_splits(
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=False,
    )
    split_loaders = {
        "train": loaded.train_loader,
        "val": loaded.val_loader,
        "test": loaded.test_loader,
    }

    stats: ChannelStats | None = None
    if normalize:
        stats_src = (
            Path(normalization_stats_path)
            if normalization_stats_path is not None
            else derive_stats_path_from_daily_hf(data_config.daily_hf_dir)
        )
        stats = ChannelStats.from_path(stats_src)
        # Drop the same stats file alongside the H5 cache so the inference
        # release bundle can copy it without ambiguity.
        copy_stats_file(stats_src, output_dir / "normalization_stats.json")
        logger.info("Using normalization stats from %s", stats_src)
        for ch in stats.channels:
            logger.info(
                "  channel %d: mean=%.4f std=%.4f",
                ch,
                float(stats.means[ch]),
                float(stats.stds[ch]),
            )

    # Per-sample artificial masker for val/test held-out scoring.
    masker = RandomNoiseMask(patch_size=10, mask_ratio=val_mask_ratio)
    mask_rng = np.random.default_rng(42)

    n_timesteps = data_config.n_days * 1440
    h5_paths: dict[str, Path] = {}

    for split in splits:
        if split not in split_loaders:
            logger.warning("Unknown split %r; skipping.", split)
            continue
        needs_ground_truth = split in ("val", "test")
        dl = split_loaders[split]
        n_samples = len(dl.dataset)
        h5_path = output_dir / f"{split}.h5"
        logger.info("Exporting %s (%d samples) → %s", split, n_samples, h5_path)

        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset(
                "X",
                shape=(n_samples, n_timesteps, 19),
                dtype="float32",
                chunks=(min(chunk_size, n_samples), n_timesteps, 19),
            )
            if needs_ground_truth:
                ds_ori = f.create_dataset(
                    "X_ori",
                    shape=(n_samples, n_timesteps, 19),
                    dtype="float32",
                    chunks=(min(chunk_size, n_samples), n_timesteps, 19),
                )

            write_idx = 0
            for batch_idx, (data, _mask) in enumerate(dl):
                batch_np = data.numpy()  # (B, C=19, T)
                if stats is not None:
                    batch_np = stats.normalize_numpy(batch_np)
                batch_transposed = np.transpose(batch_np, (0, 2, 1))  # (B, T, C)
                bsz = batch_transposed.shape[0]

                if needs_ground_truth:
                    ds_ori[write_idx : write_idx + bsz] = batch_transposed
                    masked = batch_transposed.copy()
                    for i in range(bsz):
                        sample_ct = batch_np[i]  # (C, T) normalized
                        original_mask = (~np.isnan(sample_ct)).astype(np.float64)
                        result = masker.generate(sample_ct, original_mask, mask_rng)
                        art_mask_tc = result.artificial_mask.T  # → (T, C)
                        masked[i][art_mask_tc > 0] = np.nan
                    ds[write_idx : write_idx + bsz] = masked
                else:
                    ds[write_idx : write_idx + bsz] = batch_transposed

                write_idx += bsz
                if (batch_idx + 1) % 5 == 0:
                    logger.info("  %s: wrote %d/%d samples", split, write_idx, n_samples)
            logger.info("  %s: done — exported %d samples", split, write_idx)

        h5_paths[split] = h5_path

    return h5_paths
