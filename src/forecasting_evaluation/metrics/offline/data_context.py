"""Shared offline-metrics data loading helpers aligned with evaluator data flow."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from forecasting_evaluation.data.cache_bundle import (
    prepare_history_cf_raw_cache_for_split,
)
from forecasting_evaluation.data.data_loader import ForecastingDataLoader
from forecasting_evaluation.data.online_dataset import resolve_cache_base_dir

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


def iter_offline_user_contexts_from_eval_flow(
    *,
    config,
    prediction_files: dict[str, list[Path]],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield per-user history tensors via the same split/cache flow used by evaluation."""
    data_loader = ForecastingDataLoader(config.data)
    _train_ds, _val_ds, test_ds = data_loader.load_splits()

    model_config = _resolve_cache_model_config(config, test_ds)
    _cache_dir, test_cache_path, test_row_groups = prepare_history_cf_raw_cache_for_split(
        split_name="test",
        split_ds=test_ds,
        data_config=config.data,
        model_config=model_config,
        features_config=config.features,
        h5_output_dir=str(resolve_cache_base_dir(config.data)),
        overwrite=False,
    )

    processed_users: set[str] = set()
    yielded_count = 0

    with h5py.File(test_cache_path, "r") as history_handle:
        history_rows = history_handle["history_cf_rows"]
        for row_group in test_row_groups:
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
            yielded_count += 1
            yield (
                user_id,
                {
                    "history": history,
                    "variable_names": list(row["channel_names"]),
                    "dataset_row_idx": int(row_group.dataset_row_idx),
                },
            )

    logger.info(
        "Yielded %d offline user contexts via evaluator-aligned cache flow",
        yielded_count,
    )


def load_offline_user_contexts_from_eval_flow(
    *,
    config,
    prediction_files: dict[str, list[Path]],
) -> dict[str, dict[str, Any]]:
    """Load per-user history tensors via the same split/cache flow used by evaluation."""
    user_contexts = dict(
        iter_offline_user_contexts_from_eval_flow(
            config=config,
            prediction_files=prediction_files,
        )
    )
    logger.info(
        "Loaded %d offline user contexts via evaluator-aligned cache flow",
        len(user_contexts),
    )
    return user_contexts
