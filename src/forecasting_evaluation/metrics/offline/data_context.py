"""Shared offline-metrics data loading helpers aligned with evaluator data flow."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from forecasting_evaluation.data.data_loader import ForecastingDataLoader
from forecasting_evaluation.forecasting_training.cache_bundle import (
    prepare_history_cf_cache_bundle,
)

logger = logging.getLogger(__name__)


def _resolve_cache_model_config(config, test_ds) -> SimpleNamespace:
    """Build the minimal cache config needed to materialize evaluator-style history rows."""
    if len(test_ds) > 0:
        n_features = int(np.asarray(test_ds[0]["values"]).shape[1])
    else:
        n_features = 19

    return SimpleNamespace(
        n_steps=1,
        n_pred_steps=int(config.forecasting.forecasting_length),
        n_features=n_features,
    )


def load_offline_user_contexts_from_eval_flow(
    *,
    config,
    prediction_files: dict[str, list[Path]],
) -> dict[str, dict[str, Any]]:
    """Load per-user history tensors via the same split/cache flow used by evaluation."""
    data_loader = ForecastingDataLoader(config.data)
    train_ds, val_ds, test_ds = data_loader.load_splits()

    model_config = _resolve_cache_model_config(config, test_ds)
    _cache_dir, cache_paths, row_groups_by_split, _scaler_stats = prepare_history_cf_cache_bundle(
        split_datasets={
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        },
        data_config=config.data,
        model_config=model_config,
        features_config=config.features,
        h5_output_dir="data/processed/forecasting_eval_h5",
        overwrite=False,
    )

    user_contexts: dict[str, dict[str, Any]] = {}
    processed_users: set[str] = set()

    with h5py.File(cache_paths["test"], "r") as history_handle:
        history_rows = history_handle["history_cf_rows"]
        for row_group in row_groups_by_split["test"]:
            user_id = str(row_group.user_id)
            if user_id in processed_users:
                continue
            processed_users.add(user_id)
            if user_id not in prediction_files:
                continue

            row = test_ds[int(row_group.dataset_row_idx)]
            history = np.asarray(
                history_rows[str(row_group.dataset_row_idx)][...],
                dtype=float,
            )
            user_contexts[user_id] = {
                "history": history,
                "variable_names": list(row["channel_names"]),
                "dataset_row_idx": int(row_group.dataset_row_idx),
            }

    logger.info(
        "Loaded %d offline user contexts via evaluator-aligned cache flow",
        len(user_contexts),
    )
    return user_contexts
