"""Training-pipeline orchestrator: load splits → cache bundle → fit → release.

The single public entry point is :func:`run_training`. The Hydra CLI
(``mhc-forecast-train``) is a thin wrapper around it. Library users can call it
directly with a :class:`ForecastingTrainingConfig`.

Pipeline:

1. ``seed_everything(config.seed)`` — before model construction.
2. ``ForecastingDataLoader(config.data).load_splits()`` — the SAME split logic
   the evaluator uses, so train/val are identical by construction.
3. ``prepare_history_cf_cache_bundle(...)`` — fits a train-only channel
   StandardScaler and materializes raw+standard history_cf caches + row-group
   manifests for all splits (cache hit reuses them).
4. ``build_pypots_forecasting_dataset(...)`` for train and val, from the
   standardized cache (when ``whether_standardscaler``) with short-history
   windows included (NaN-left-padded, default ON).
5. ``create_model(...)`` then a custom training loop (see below).
6. ``write_release(...)`` — packages the checkpoint, scaler stats, full
   training_config.json, and the openmhc manifest into a release bundle the
   ``mhc-forecast-eval`` CLI consumes via ``model.release_dir=``.

Why a custom training loop instead of ``model.fit``: PyPOTS' ``fit`` re-wraps
``train_set`` in a stock ``BaseDataset``, discarding our manifest-backed
``PyPOTSForecastingDataset`` (which slices windows on demand, NaN-left-pads
short histories, and keeps a row-grouped batch sampler). So we build
``DataLoader``s from our dataset and drive the model's own ``_train_model`` —
the exact internal entry ``fit`` calls — then save the same way ``fit`` does.
The optimizer and TensorBoard ``summary_writer`` are already set up in the model
constructor (we pass ``saving_path``), so W&B ``sync_tensorboard`` still works.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from forecasting_evaluation.data.cache_bundle import prepare_history_cf_cache_bundle
from forecasting_evaluation.data.data_loader import ForecastingDataLoader
from forecasting_evaluation.data.online_dataset import (
    build_pypots_forecasting_dataset,
    resolve_cache_base_dir,
)
from forecasting_training.config import ForecastingTrainingConfig
from forecasting_training.model_registry import create_model
from forecasting_training.release import write_release
from forecasting_training.seeding import seed_everything

logger = logging.getLogger(__name__)


def _maybe_init_wandb(config: ForecastingTrainingConfig) -> Any | None:
    """Initialize a W&B run if ``output.wandb_enabled`` is true.

    Uses ``sync_tensorboard=True`` so PyPOTS' own TensorBoard scalars stream into
    the W&B run with no extra logging hooks. Authentication comes from
    ``WANDB_API_KEY`` or ``~/.netrc``. Failure to init is logged, never raised.
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
    """Locate the ``.pypots`` checkpoint PyPOTS just wrote under ``saving_path``."""
    base = Path(saving_path)
    if not base.exists():
        raise FileNotFoundError(f"PyPOTS saving_path missing: {base}")
    candidates = [p for p in base.rglob("*.pypots") if not p.name.startswith("events.out.tfevents")]
    if not candidates:
        raise FileNotFoundError(f"No .pypots files under {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _make_loader(dataset, batch_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    """Build a DataLoader using the dataset's row-grouped batch sampler."""
    return DataLoader(
        dataset,
        batch_sampler=dataset.build_batch_sampler(batch_size, shuffle=shuffle),
        num_workers=num_workers,
    )


def run_training(config: ForecastingTrainingConfig) -> Path:
    """Train a PyPOTS forecaster end-to-end and return the release-bundle path."""
    # ---- 1. Seed (before model construction) ----------------------------
    seed_everything(config.seed)

    # ---- 1.5. W&B (before fit so sync_tensorboard catches epoch-1 scalars)
    wandb_run = _maybe_init_wandb(config)

    # ---- 2. Splits (identical to eval) ----------------------------------
    train_ds, val_ds, test_ds = ForecastingDataLoader(config.data).load_splits()
    split_datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    # ---- 3. History_cf cache bundle (train-fit scaler + manifests) ------
    _cache_dir, cache_paths, row_groups_by_split, _scaler_stats = prepare_history_cf_cache_bundle(
        split_datasets=split_datasets,
        data_config=config.data,
        model_config=config.model,
        features_config=config.features,
        h5_output_dir=resolve_cache_base_dir(config.data),
    )

    # ---- 4. Train/val datasets (standardized when configured) -----------
    standardized = config.training.whether_standardscaler
    train_source = cache_paths["train_standard" if standardized else "train"]
    val_source = cache_paths["val_standard" if standardized else "val"]
    offset = int(config.forecasting.daily_start_hour_offset)
    include_short = config.training.include_short_history

    train_dataset = build_pypots_forecasting_dataset(
        split_ds=train_ds,
        sample_index_file=config.data.sample_index_file,
        model_config=config.model,
        features_config=config.features,
        daily_start_hour_offset=offset,
        history_cf_source=train_source,
        row_groups=row_groups_by_split["train"],
        include_short_history=include_short,
    )
    val_dataset = build_pypots_forecasting_dataset(
        split_ds=val_ds,
        sample_index_file=config.data.sample_index_file,
        model_config=config.model,
        features_config=config.features,
        daily_start_hour_offset=offset,
        history_cf_source=val_source,
        row_groups=row_groups_by_split["val"],
        include_short_history=include_short,
    )
    logger.info(
        "Datasets ready: train=%d samples, val=%d samples (standardized=%s, include_short=%s)",
        len(train_dataset),
        len(val_dataset),
        standardized,
        include_short,
    )

    # ---- 5. Model + custom training loop --------------------------------
    model = create_model(config.model, config.training, config.output)
    train_loader = _make_loader(
        train_dataset,
        config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = _make_loader(
        val_dataset,
        config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    logger.info(
        "Fitting %s for up to %d epochs on %s",
        config.model.model_name,
        config.training.epochs,
        config.training.device,
    )
    model._train_model(train_loader, val_loader)
    model.model.load_state_dict(model.best_model_dict)
    model._auto_save_model_if_necessary(
        confirm_saving=config.output.model_saving_strategy == "best"
    )
    logger.info("Training complete.")

    # ---- 6. Release bundle ----------------------------------------------
    ckpt = _find_trained_checkpoint(config.output.saving_path)
    logger.info("Selected checkpoint for release: %s", ckpt)

    release_dir = (
        Path(config.output.release_dir)
        if config.output.release_dir is not None
        else Path(config.output.saving_path).with_name(
            Path(config.output.saving_path).name + "_release"
        )
    )
    scaler_stats_path = cache_paths["scaler_stats"] if standardized else None

    bundle = write_release(
        model_name=config.model.model_name,
        arch={
            "n_steps": config.model.n_steps,
            "n_pred_steps": config.model.n_pred_steps,
            "n_features": config.model.n_features,
        },
        training_config_json=asdict(config),
        release_dir=release_dir,
        pypots_checkpoint=ckpt,
        scaler_stats_path=scaler_stats_path,
        provenance={
            "seed": config.seed,
            "model_name": config.model.model_name,
            "training": asdict(config.training),
            "checkpoint_source": str(ckpt),
        },
    )
    logger.info("Done. Release bundle: %s", bundle)

    # ---- 7. W&B finish --------------------------------------------------
    if wandb_run is not None:
        try:
            wandb_run.summary["release_dir"] = str(bundle)
            if config.output.upload_wandb_artifact:
                import wandb

                artifact = wandb.Artifact(
                    name=f"forecasting-{config.model.model_name}-release",
                    type="model",
                )
                artifact.add_dir(str(bundle))
                wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb finalization failed (%s); ignoring.", exc)

    return bundle
