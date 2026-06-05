"""Main orchestrator for forecasting evaluation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import h5py
import numpy as np

from forecasting_evaluation.config import print_config
from forecasting_evaluation.data.data_loader import ForecastingDataLoader
from forecasting_evaluation.data.types import SubTrajectoryInput
from forecasting_evaluation.forecasting_training.cache_bundle import (
    prepare_history_cf_cache_bundle,
    prepare_history_cf_raw_cache_for_split,
)
from forecasting_evaluation.forecasting_training.online_dataset import _resolve_window_hours
from forecasting_evaluation.io.predict_result_writer import PredictResultWriter, PublicWriter
from forecasting_evaluation.models.deep_learning_model.pypots_forecasting_base import (
    BasePyPOTSForecastingModel,
)
from forecasting_evaluation.models.registry import create_forecasting_model

if TYPE_CHECKING:
    from forecasting_evaluation.config import ForecastingEvalConfig
    from forecasting_evaluation.models.base import BasePredictionModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationDataContext:
    """Unified evaluation-time data context prepared before model execution."""

    train_ds: object
    val_ds: object
    test_ds: object
    test_cache_path: Path | None = None
    test_row_groups: list | None = None


class ForecastingEvaluator:
    """Main orchestrator for prediction generation and persistence."""

    def __init__(self, config: ForecastingEvalConfig):
        """Initialize evaluator.

        Args:
            config: Full forecasting evaluation configuration.
        """
        self.config = config

        # Set random seeds for reproducibility
        np.random.seed(config.seed)
        random.seed(config.seed)

    def run(self) -> dict:
        """Execute full evaluation pipeline.

        Returns:
            Run metadata with output path and saved sample count.
        """
        # 0) Log resolved config first so output artifacts can always be traced.
        print_config(self.config)

        model: BasePredictionModel = create_forecasting_model(
            self.config.model,
            seed=self.config.seed,
            forecasting_config=self.config.forecasting,
            features_config=self.config.features,
        )

        # 1) Build a unified evaluation context (splits + cache + row groups).
        data_context = self._load_evaluation_data(model)

        # 2) Initialize run-level writer for shared metadata/config persistence.
        public_writer = PublicWriter(
            self.config,
            experiment_name=self.config.experiment_name,
        )

        # 3) Execute one model sequentially over all prepared test trajectories.
        self._run_sequential(model, data_context, public_writer)

        # 4) Flush run-level metadata and return a compact run summary.
        run_dir = public_writer.finalize()
        return {
            "run_dir": str(run_dir),
            "prediction_samples": public_writer.total_written,
            "skipped_users": public_writer.skipped_users,
        }

    def _load_evaluation_data(self, model: BasePredictionModel) -> EvaluationDataContext:
        """Load all data needed by the current model during the unified load step."""
        data_loader = ForecastingDataLoader(self.config.data)
        train_ds, val_ds, test_ds = data_loader.load_splits()

        context = EvaluationDataContext(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
        )
        # Resolve cache layout once so all model types use one loader path.
        model_config, h5_output_dir = self._resolve_eval_cache_config(model, test_ds)
        if not isinstance(model, BasePyPOTSForecastingModel):
            _cache_dir, test_cache_path, test_row_groups = prepare_history_cf_raw_cache_for_split(
                split_name="test",
                split_ds=test_ds,
                data_config=self.config.data,
                model_config=model_config,
                features_config=self.config.features,
                h5_output_dir=h5_output_dir,
                overwrite=False,
            )
            context.test_cache_path = test_cache_path
            context.test_row_groups = test_row_groups
            logger.info(
                "Prepared eval dataset using raw cache at %s",
                context.test_cache_path,
            )
            return context

        split_datasets = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }
        # Cache bundle generation is split-aware: train/val/test are prepared together
        # even though prediction is only emitted for test trajectories.
        _cache_dir, cache_paths, row_groups_by_split, _scaler_stats = prepare_history_cf_cache_bundle(
            split_datasets=split_datasets,
            data_config=self.config.data,
            model_config=model_config,
            features_config=self.config.features,
            h5_output_dir=h5_output_dir,
            overwrite=False,
            scaler_stats_override=(
                model.scaler_stats if isinstance(model, BasePyPOTSForecastingModel) else None
            ),
        )
        context.test_cache_path = (
            cache_paths["test_standard"]
            if isinstance(model, BasePyPOTSForecastingModel) and model.uses_standard_scaler
            else cache_paths["test"]
        )
        context.test_row_groups = row_groups_by_split["test"]
        logger.info(
            "Prepared eval dataset using %s cache at %s",
            (
                "standardized"
                if isinstance(model, BasePyPOTSForecastingModel) and model.uses_standard_scaler
                else "raw"
            ),
            context.test_cache_path,
        )
        return context

    def _resolve_eval_cache_config(self, model: BasePredictionModel, test_ds) -> tuple[SimpleNamespace, str]:
        """Resolve one cache-bundle config so every model can share the same loader path."""
        if isinstance(model, BasePyPOTSForecastingModel):
            h5_output_dir = model._get_training_config_value("h5_export", "output_dir")
            if not h5_output_dir:
                h5_output_dir = "data/processed/forecasting_pypots_h5"
            if model.uses_standard_scaler and model.scaler_stats is None:
                raise FileNotFoundError(
                    "PyPOTS checkpoint was trained with standard scaling, but "
                    "standard_scaler_stats.json could not be resolved from the saved training config."
                )
            return (
                SimpleNamespace(
                    n_steps=int(model.n_steps),
                    n_pred_steps=int(model.n_pred_steps),
                    n_features=int(model.n_features),
                ),
                h5_output_dir,
            )

        if len(test_ds) > 0:
            n_features = int(np.asarray(test_ds[0]["values"]).shape[1])
        else:
            # Keep a stable fallback for empty test splits.
            n_features = 19
        return (
            SimpleNamespace(
                n_steps=1,
                n_pred_steps=int(self.config.forecasting.forecasting_length),
                n_features=n_features,
            ),
            "data/processed/forecasting_eval_h5",
        )

    def _run_sequential(
        self,
        model: BasePredictionModel,
        data_context: EvaluationDataContext,
        public_writer: PublicWriter,
    ) -> None:
        """Run current sequential evaluation path with one shared cached sample loader."""
        model_name = getattr(model, "model_name", model.__class__.__name__)
        model_dir = public_writer.prepare_model_dir(model_name)
        test_ds = data_context.test_ds
        if data_context.test_cache_path is None or data_context.test_row_groups is None:
            raise RuntimeError("Evaluation data context is incomplete")

        logger.info(
            "Running model %s on %d trajectories with shared cached sample loader",
            model_name,
            len(test_ds),
        )
        daily_start_hour_offset = self._resolve_runtime_daily_start_hour_offset(model)
        prediction_hours = (
            int(model.n_pred_steps)
            if isinstance(model, BasePyPOTSForecastingModel)
            else int(self.config.forecasting.forecasting_length)
        )

        with h5py.File(data_context.test_cache_path, "r") as history_handle:
            for row_group in data_context.test_row_groups:
                user_id = str(row_group.user_id)
                if public_writer.should_skip_user(user_id):
                    logger.info(
                        "[%s] Skip user_id=%s because parquet exists and overwrite is disabled",
                        model_name,
                        user_id,
                    )
                    public_writer.increment_skipped_users()
                    continue

                row = test_ds[int(row_group.dataset_row_idx)]
                logger.info(
                    "[%s] Processing cached trajectory row %d/%d (user_id: %s), length: %d",
                    model_name,
                    int(row_group.dataset_row_idx) + 1,
                    len(test_ds),
                    user_id,
                    len(row["values"]),
                )
                history_cf = np.asarray(
                    history_handle["history_cf_rows"][str(row_group.dataset_row_idx)][...],
                    dtype=np.float32,
                )
                predict_writer = PredictResultWriter(
                    model_dir=model_dir,
                    user_id=user_id,
                    overwrite_existing=self.config.output.overwrite_existing_parquet,
                )

                try:
                    for window in row_group.windows:
                        # Runtime hour boundaries are derived from day index + offset.
                        history_end_hour, pred_end_hour = _resolve_window_hours(
                            current_day=int(window.current_day),
                            horizon_length=prediction_hours,
                            daily_start_hour_offset=daily_start_hour_offset,
                        )

                        if history_end_hour <= 0:
                            continue

                        if (
                            isinstance(model, BasePyPOTSForecastingModel)
                            and history_end_hour < int(model.n_steps)
                        ):
                            continue

                        if pred_end_hour > history_cf.shape[1]:
                            continue

                        # History slicing depends on model family:
                        # - PyPOTS: fixed-length trailing context.
                        # - Others: full-prefix context up to forecast origin.
                        history_window = self._build_history_window(
                            model=model,
                            history_cf=history_cf,
                            history_end_hour=history_end_hour,
                        )

                        if history_window.shape[1] <= 0:
                            continue

                        # Target window is always the absolute horizon slice from origin.
                        target_window = history_cf[
                            :,
                            history_end_hour:pred_end_hour,
                        ]

                        sub_trajectory = SubTrajectoryInput(
                            history=history_window,
                            variable_names=list(row["channel_names"]),
                            past_covariates={},
                            future_covariates={},
                            static_covariates=None,
                            ground_truth=target_window,
                            index_days=int(window.current_day),
                            prediction_hours=prediction_hours,
                        )

                        point_result, quantiles_result, base_result = model.predict_wrapper(
                            sub_trajectory
                        )

                        prediction_record = {
                            "user_id": user_id,
                            "model": model_name,
                            "history_length": int(history_end_hour),
                            "point_predictions": point_result,
                            "quantile_predictions": quantiles_result,
                            "performance": base_result,
                        }
                        quantile_levels = getattr(model, "quantile_levels", None)
                        if quantiles_result is not None and quantile_levels is not None:
                            prediction_record["quantile_levels"] = quantile_levels

                        predict_writer.append(prediction_record)
                finally:
                    predict_writer.close()
                    # Reset model state between users to avoid cross-user leakage.
                    model.reset()

                public_writer.increment_written(predict_writer.records_written)

    def _resolve_runtime_daily_start_hour_offset(self, model: BasePredictionModel) -> int:
        """Resolve the effective runtime offset for one evaluation run."""
        eval_offset = int(self.config.forecasting.daily_start_hour_offset)
        eval_offset_explicit = bool(
            getattr(self.config.forecasting, "_daily_start_hour_offset_explicit", False)
        )

        if not isinstance(model, BasePyPOTSForecastingModel):
            return eval_offset

        training_offset = int(model.training_daily_start_hour_offset)
        # For fixed-window deep models, day-boundary mismatch changes samples.
        # Enforce exact match when eval config explicitly sets an offset.
        if eval_offset_explicit and eval_offset != training_offset:
            raise ValueError(
                "Eval config forecasting.daily_start_hour_offset="
                f"{eval_offset} does not match checkpoint training offset "
                f"{training_offset}."
            )
        return training_offset

    def _build_history_window(
        self,
        *,
        model: BasePredictionModel,
        history_cf: np.ndarray,
        history_end_hour: int,
    ) -> np.ndarray:
        """Slice one model-ready history window from the shared cached row tensor."""
        if isinstance(model, BasePyPOTSForecastingModel):
            return history_cf[:, history_end_hour - model.n_steps : history_end_hour]
        return history_cf[:, :history_end_hour]
