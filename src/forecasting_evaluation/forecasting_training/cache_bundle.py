"""Shared history_cf cache bundle orchestration for training and evaluation."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import yaml

from forecasting_evaluation.forecasting_training.online_dataset import (
    build_history_cf_rows,
    history_cf_cache_subdir,
    history_cf_manifest_path,
    load_or_build_row_group_manifest,
    write_history_cf_cache,
)
from forecasting_evaluation.forecasting_training.standard_scaler import (
    ChannelStandardScalerStats,
    fit_from_history_cf_rows,
)

logger = logging.getLogger(__name__)


def save_data_config_yaml(cache_dir: Path, data_config) -> Path:
    """Persist the effective data-layer config beside the generated caches."""
    output_path = cache_dir / "config.yaml"
    payload = asdict(data_config)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return output_path


def cache_bundle_paths(cache_dir: Path) -> dict[str, Path]:
    """Return the fixed set of cache bundle paths under one cache directory."""
    return {
        "train": cache_dir / "train.h5",
        "train_standard": cache_dir / "train_standard.h5",
        "val": cache_dir / "val.h5",
        "val_standard": cache_dir / "val_standard.h5",
        "test": cache_dir / "test.h5",
        "test_standard": cache_dir / "test_standard.h5",
        "data_config": cache_dir / "config.yaml",
        "scaler_stats": cache_dir / "standard_scaler_stats.json",
        "train_manifest": cache_dir / "train_manifest.json",
        "val_manifest": cache_dir / "val_manifest.json",
        "test_manifest": cache_dir / "test_manifest.json",
    }


def prepare_history_cf_cache_bundle(
    *,
    split_datasets: dict[str, object],
    data_config,
    model_config,
    features_config,
    h5_output_dir: str | Path,
    overwrite: bool = False,
    scaler_stats_override: ChannelStandardScalerStats | None = None,
) -> tuple[Path, dict[str, Path], dict[str, list], ChannelStandardScalerStats]:
    """Ensure the raw+standard history_cf cache bundle exists for all three splits."""
    cache_dir = history_cf_cache_subdir(
        base_dir=Path(h5_output_dir) / "history_cf_cache",
        data_config=data_config,
        model_config=model_config,
        features_config=features_config,
    )
    cache_paths = cache_bundle_paths(cache_dir)
    cache_bundle_ready = not overwrite and all(path.exists() for path in cache_paths.values())

    row_groups_by_split: dict[str, list] = {}
    scaler_stats: ChannelStandardScalerStats | None = scaler_stats_override

    if cache_bundle_ready:
        logger.info("history_cf cache bundle already exists under %s, reusing it", cache_dir)
        for split_name, split_ds in split_datasets.items():
            row_groups_by_split[split_name] = load_or_build_row_group_manifest(
                split_ds=split_ds,
                sample_index_file=data_config.sample_index_file,
                model_config=model_config,
                manifest_path=history_cf_manifest_path(cache_dir, split_name),
                split_name=split_name,
                overwrite=False,
            )
        if scaler_stats is None:
            from forecasting_evaluation.forecasting_training.standard_scaler import load_stats_json

            scaler_stats = load_stats_json(cache_paths["scaler_stats"])
    else:
        logger.info("Building history_cf cache bundle under %s", cache_dir)
        rows_by_split: dict[str, list] = {}
        for split_name, split_ds in split_datasets.items():
            logger.info("Building history_cf rows for %s split", split_name)
            rows_by_split[split_name] = build_history_cf_rows(
                split_ds=split_ds,
                features_config=features_config,
                model_config=model_config,
            )

        if scaler_stats is None:
            logger.info("Fitting StandardScaler stats from train split history_cf rows")
            scaler_stats = fit_from_history_cf_rows(
                rows_by_split["train"],
                n_channels=model_config.n_features,
            )
        else:
            logger.info("Using provided StandardScaler stats to materialize standard caches")

        scaler_stats_path = scaler_stats.save_stats_json(cache_paths["scaler_stats"])
        data_config_path = save_data_config_yaml(cache_dir, data_config)

        for split_name, history_rows in rows_by_split.items():
            write_history_cf_cache(
                history_rows,
                cache_paths[split_name],
                overwrite=True,
            )
            write_history_cf_cache(
                [scaler_stats.transform_history_cf(row) for row in history_rows],
                cache_paths[f"{split_name}_standard"],
                overwrite=True,
            )

        for split_name, split_ds in split_datasets.items():
            row_groups_by_split[split_name] = load_or_build_row_group_manifest(
                split_ds=split_ds,
                sample_index_file=data_config.sample_index_file,
                model_config=model_config,
                manifest_path=history_cf_manifest_path(cache_dir, split_name),
                split_name=split_name,
                overwrite=True,
            )

        logger.info("Saved scaler stats to %s", scaler_stats_path)
        logger.info("Saved cache data config to %s", data_config_path)

    logger.info(
        "Prepared history_cf cache bundle under %s", cache_dir
    )
    logger.info(
        "Cache files ready: train=%s, train_standard=%s, val=%s, val_standard=%s, test=%s, test_standard=%s, train_manifest=%s, val_manifest=%s, test_manifest=%s, data_config=%s, scaler_stats=%s",
        cache_paths["train"],
        cache_paths["train_standard"],
        cache_paths["val"],
        cache_paths["val_standard"],
        cache_paths["test"],
        cache_paths["test_standard"],
        cache_paths["train_manifest"],
        cache_paths["val_manifest"],
        cache_paths["test_manifest"],
        cache_paths["data_config"],
        cache_paths["scaler_stats"],
    )
    logger.info(
        "Manifest row-groups ready: train=%d rows/%d samples, val=%d rows/%d samples, test=%d rows/%d samples",
        len(row_groups_by_split["train"]),
        sum(len(group.windows) for group in row_groups_by_split["train"]),
        len(row_groups_by_split["val"]),
        sum(len(group.windows) for group in row_groups_by_split["val"]),
        len(row_groups_by_split["test"]),
        sum(len(group.windows) for group in row_groups_by_split["test"]),
    )

    if scaler_stats is None:
        raise RuntimeError("Scaler stats should be available after preparing the cache bundle")

    return cache_dir, cache_paths, row_groups_by_split, scaler_stats
