"""Training-pipeline orchestrator: data export → model fit → release bundle.

The single public entry point is :func:`run_training`. The Hydra CLI
(``mhc-impute-train``) is a thin wrapper around it. Library users
(notebooks, sweep runners, integration tests) can call it directly
with a :class:`PyPOTSTrainingConfig` instance.

The orchestration is intentionally linear and easy to read:

1. ``seed_everything(config.seed)`` — must come before model construction
   so FEDformer's ``FourierBlock.__init__`` draws indices deterministically.
2. ``export_splits_to_h5(...)`` — content-addressed H5 cache (skipped if
   files already exist for this config).
3. ``create_model(...)`` — instantiates the PyPOTS model with random
   weights.
4. ``model.fit(...)`` — PyPOTS' own training loop.
5. ``write_release(...)`` — packages the trained checkpoint into an
   OpenMHC release bundle (manifest + checkpoint + normalization +
   optional Fourier-modes sidecar).

The bundle path is returned so callers can plug it into the eval CLI's
``method.release_dir=...`` flag directly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from imputation_training.config import PyPOTSTrainingConfig
from imputation_training.data_export import (
    export_splits_to_h5,
    h5_cache_subdir,
    h5_files_exist,
)
from imputation_training.model_registry import create_model
from imputation_training.release import write_release
from imputation_training.seeding import seed_everything

logger = logging.getLogger(__name__)


def _maybe_init_wandb(config: PyPOTSTrainingConfig) -> Any | None:
    """Initialize a W&B run if ``output.wandb_enabled`` is true.

    Uses ``sync_tensorboard=True`` so PyPOTS' own TensorBoard scalars
    (train/MAE, validating/MAE) stream into the W&B run with no extra
    logging hooks. Authentication comes from ``WANDB_API_KEY`` or
    ``~/.netrc`` — the caller is responsible for one of those being set.

    Returns the wandb ``Run`` (or ``None`` if disabled / library missing).
    Failure to init is logged but never raised — training is the
    important thing; observability is best-effort.
    """
    if not config.output.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("output.wandb_enabled=true but `wandb` not installed; skipping.")
        return None
    try:
        run = wandb.init(
            project=config.output.wandb_project,
            entity=config.output.wandb_entity,
            config=asdict(config),
            sync_tensorboard=True,
        )
    except Exception as exc:  # noqa: BLE001 - wandb raises various error types
        logger.warning("wandb.init failed (%s); continuing without W&B.", exc)
        return None
    logger.info("W&B run: %s", run.url)
    return run


def _find_trained_checkpoint(saving_path: str | Path) -> Path:
    """Locate the ``.pypots`` checkpoint PyPOTS just wrote.

    PyPOTS writes to ``<saving_path>/<YYYYMMDD_THHMMSS>/<MODEL>.pypots``
    and also drops TensorBoard event files (also named ``*.pypots``)
    under a ``tensorboard/`` sibling. We pick the most-recent run dir
    and return its model file.
    """
    base = Path(saving_path)
    if not base.exists():
        raise FileNotFoundError(f"PyPOTS saving_path missing: {base}")
    candidates = [p for p in base.rglob("*.pypots") if not p.name.startswith("events.out.tfevents")]
    if not candidates:
        raise FileNotFoundError(f"No .pypots files under {base}")
    # Sort by mtime; most-recent first.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_training(config: PyPOTSTrainingConfig) -> Path:
    """Train a PyPOTS imputer end-to-end and return the release-bundle path.

    See module docstring for the pipeline steps.

    Args:
        config: Full training configuration.

    Returns:
        Path to the OpenMHC release bundle (``release_dir``). The
        directory contains ``model.pypots``, ``normalization_stats.json``,
        ``openmhc_manifest.json``, and (for FEDformer)
        ``fourier_modes.json``.
    """
    # ---- 1. Seed --------------------------------------------------------
    seed_everything(config.seed)

    # ---- 1.5. W&B (must come before model creation so sync_tensorboard
    # ----       picks up PyPOTS' TensorBoard scalars from epoch 1). -----
    wandb_run = _maybe_init_wandb(config)

    # ---- 2. H5 export ---------------------------------------------------
    h5_dir = h5_cache_subdir(config.h5_export.output_dir, config.data, config.h5_export)
    h5_dir.mkdir(parents=True, exist_ok=True)
    if h5_files_exist(h5_dir) and not config.h5_export.overwrite:
        logger.info("Skipping H5 export (cache hit): %s", h5_dir)
    else:
        logger.info("Exporting splits to H5: %s", h5_dir)
    h5_paths = export_splits_to_h5(
        config.data,
        h5_dir,
        chunk_size=config.h5_export.chunk_size,
        overwrite=config.h5_export.overwrite,
        val_mask_ratio=config.h5_export.val_mask_ratio,
        normalize=config.h5_export.normalize,
    )

    # ---- 3. Model -------------------------------------------------------
    # Resolve "auto" → concrete device once so PyPOTS (which does not
    # understand "auto") sees a real device string.
    from openmhc._device import resolve_device

    config.training.device = resolve_device(config.training.device)
    model = create_model(config.model, config.training, config.output)

    # ---- 4. Fit ---------------------------------------------------------
    logger.info(
        "Fitting %s for up to %d epochs on %s",
        config.model.model_name,
        config.training.epochs,
        config.training.device,
    )
    model.fit(
        train_set=str(h5_paths["train"]),
        val_set=str(h5_paths["val"]),
        file_type="hdf5",
    )
    logger.info("Training complete.")

    # ---- 5. Release bundle ---------------------------------------------
    ckpt = _find_trained_checkpoint(config.output.saving_path)
    logger.info("Selected checkpoint for release: %s", ckpt)

    # Default release dir: a sibling of saving_path with a "_release" suffix.
    release_dir = (
        Path(config.output.release_dir)
        if config.output.release_dir is not None
        else Path(config.output.saving_path).with_name(
            Path(config.output.saving_path).name + "_release"
        )
    )

    # Use the stats file that the H5 export materialized alongside the
    # H5 cache — it's the same file inference will read.
    stats_path: Path | None = None
    if config.h5_export.normalize:
        stats_path = h5_dir / "normalization_stats.json"
        if not stats_path.exists():
            stats_path = None  # defensive — should never happen

    bundle = write_release(
        model=model,
        model_config=config.model,
        release_dir=release_dir,
        pypots_checkpoint=ckpt,
        normalization_stats=stats_path,
        provenance={
            "seed": config.seed,
            "training": asdict(config.training),
            "model_config": asdict(config.model),
            "h5_cache_dir": str(h5_dir),
            "checkpoint_source": str(ckpt),
        },
    )
    logger.info("Done. Release bundle: %s", bundle)

    # ---- 6. W&B finish --------------------------------------------------
    if wandb_run is not None:
        try:
            wandb_run.summary["release_dir"] = str(bundle)
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb.finish failed (%s); ignoring.", exc)

    return bundle
