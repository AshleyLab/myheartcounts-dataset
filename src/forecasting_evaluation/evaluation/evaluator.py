"""Main orchestrator for forecasting evaluation."""

from __future__ import annotations

import inspect
import logging
import random
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import h5py
import numpy as np

from forecasting_evaluation.config import print_config
from forecasting_evaluation.data.data_loader import ForecastingDataLoader
from forecasting_evaluation.forecasting_training.cache_bundle import (
    prepare_history_cf_raw_cache_for_split,
)
from forecasting_evaluation.forecasting_training.online_dataset import (
    _resolve_window_hours,
    resolve_cache_base_dir,
)
from forecasting_evaluation.io.predict_result_writer import PredictResultWriter, PublicWriter
from forecasting_evaluation.models.naive.seasonal_naive import SeasonalNaiveModel
from forecasting_evaluation.models.registry import create_forecasting_model

if TYPE_CHECKING:
    from forecasting_evaluation.config import ForecastingEvalConfig
    from forecasting_evaluation.models.base import BasePredictionModel

logger = logging.getLogger(__name__)

# Optional metadata kwargs the harness can forward to a model's predict(), if
# the model declares them. The harness inspects each model's signature once and
# forwards only the declared subset (the same duck-typed pattern as Encoder /
# Imputer protocols).
_OPTIONAL_PREDICT_KWARGS = (
    "variable_names",
    "past_covariates",
    "future_covariates",
    "index_days",
)

# Per-class cache of the optional kwargs a model's predict() accepts.
_PREDICT_KWARG_CACHE: dict[type, set[str]] = {}


def _declared_optional_kwargs(predict_fn) -> set[str]:
    """Return the subset of optional metadata kwargs a predict() accepts.

    A ``**kwargs`` in the signature means "accepts all".
    """
    params = inspect.signature(predict_fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return set(_OPTIONAL_PREDICT_KWARGS)
    declared = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return declared & set(_OPTIONAL_PREDICT_KWARGS)


def _forward_kwargs(model, candidates: dict) -> dict:
    """Select only the optional metadata kwargs the model's predict() declares."""
    cls = type(model)
    allowed = _PREDICT_KWARG_CACHE.get(cls)
    if allowed is None:
        allowed = _declared_optional_kwargs(model.predict)
        _PREDICT_KWARG_CACHE[cls] = allowed
    return {k: v for k, v in candidates.items() if k in allowed}


def _normalize_forecast_output(output) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Normalize a model's return to ``(point, quantiles)``.

    Accepts either a bare point array or a ``(point, quantiles)`` tuple.
    """
    if isinstance(output, tuple):
        point = output[0]
        quantiles = output[1] if len(output) > 1 else None
    else:
        point, quantiles = output, None
    if point is not None:
        point = np.asarray(point, dtype=np.float32)
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float32)
    return point, quantiles


def _invoke_forecaster(
    model,
    history: np.ndarray,
    horizon: int,
    meta: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Call ``model.predict`` under the unified contract with perf tracking.

    Forwards only the optional metadata kwargs the model declares, times the
    call, tracks peak memory, and normalizes the return to
    ``(point, quantiles, perf)``.
    """
    kwargs = _forward_kwargs(model, meta)
    tracemalloc.start()
    start_time = time.time()
    output = model.predict(history, horizon, **kwargs)
    prediction_time = time.time() - start_time
    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    point_result, quantiles_result = _normalize_forecast_output(output)
    perf = {
        "prediction_time_seconds": float(prediction_time),
        "memory_usage_mb": float(peak_mem / 1024 / 1024),
    }
    return point_result, quantiles_result, perf


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

        # Populated by ``_run_sequential``; always present so ``run`` can read it.
        self._fallback_summary: dict = {"overall_fallback_rate": 0.0, "fallback_rate": {}}

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
        data_context = self._load_evaluation_data()

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
            **self._fallback_summary,
        }

    def _load_evaluation_data(self) -> EvaluationDataContext:
        """Load all data needed during evaluation via one model-agnostic path.

        Every model reads the same raw full-trajectory history cache and the same
        data-quality-only window manifest, so the in-scope window set is identical
        across model families. Models that need a fixed context window slice it
        themselves; models trained on standardized inputs standardize internally.
        """
        data_loader = ForecastingDataLoader(self.config.data)
        train_ds, val_ds, test_ds = data_loader.load_splits()

        context = EvaluationDataContext(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
        )
        model_config, h5_output_dir = self._resolve_eval_cache_config(test_ds)
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
        logger.info("Prepared eval dataset using raw cache at %s", context.test_cache_path)
        return context

    def _resolve_eval_cache_config(self, test_ds) -> tuple[SimpleNamespace, str]:
        """Resolve one model-agnostic cache-bundle config shared by all models.

        The eval cache holds raw full-trajectory history and a data-quality-only
        window manifest (``n_steps=1`` ⇒ no model-capability filtering). Models
        that need a fixed context window slice it themselves from the full prefix.
        """
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
            str(resolve_cache_base_dir(self.config.data)),
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
        prediction_hours = int(self.config.forecasting.forecasting_length)
        model_pred_steps = getattr(model, "n_pred_steps", None)
        if model_pred_steps is not None and int(model_pred_steps) != prediction_hours:
            raise ValueError(
                f"Model exposes n_pred_steps={int(model_pred_steps)} but eval "
                f"forecasting_length={prediction_hours}; a horizon mismatch would "
                "desync the prediction window from the metrics path."
            )

        # Seasonal-Naive baseline used to fill any NaN the model emits, so every
        # in-scope forecast cell is scored. Deterministic + model-agnostic.
        fallback_model = SeasonalNaiveModel(seed=self.config.seed, seasonal=24)
        fallback_substituted: np.ndarray | None = None
        fallback_total: np.ndarray | None = None

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
                n_channels = int(history_cf.shape[0])
                if fallback_substituted is None:
                    fallback_substituted = np.zeros(n_channels, dtype=np.int64)
                    fallback_total = np.zeros(n_channels, dtype=np.int64)
                variable_names = list(row["channel_names"])
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

                        # Data-quality drops only (applied equally to every model):
                        # no history before the day boundary, and ground truth must
                        # exist for the full horizon. No model-capability drops.
                        if history_end_hour <= 0:
                            continue
                        if pred_end_hour > history_cf.shape[1]:
                            continue

                        # Every model receives the full raw history prefix; it owns
                        # any context windowing / truncation / padding it needs.
                        history_window = history_cf[:, :history_end_hour]

                        # Target window is always the absolute horizon slice from origin.
                        target_window = history_cf[:, history_end_hour:pred_end_hour]

                        meta = {
                            "variable_names": variable_names,
                            "past_covariates": None,
                            "future_covariates": None,
                            "index_days": int(window.current_day),
                        }
                        point_result, quantiles_result, base_result = _invoke_forecaster(
                            model,
                            history_window,
                            prediction_hours,
                            meta,
                        )

                        # Fill NaN predictions with the Seasonal-Naive baseline so
                        # gaps are scored (not silently dropped) and record where.
                        point_result, fallback_mask = self._apply_seasonal_naive_fallback(
                            point_result=point_result,
                            history_window=history_window,
                            horizon=prediction_hours,
                            n_channels=n_channels,
                            fallback_model=fallback_model,
                        )
                        fallback_substituted += fallback_mask.sum(axis=1)
                        fallback_total += prediction_hours

                        prediction_record = {
                            "user_id": user_id,
                            "model": model_name,
                            "history_length": int(history_end_hour),
                            "point_predictions": point_result,
                            "quantile_predictions": quantiles_result,
                            # Boolean mask (n_channels, horizon): True where the model
                            # emitted NaN and the Seasonal-Naive baseline was substituted.
                            "fallback_mask": fallback_mask,
                            # Raw ground-truth slice (n_channels, horizon) co-located with the
                            # predictions so post-hoc metric recomputation / bootstrapping needs no
                            # model re-run. Observed mask is recoverable as isfinite(ground_truth).
                            "ground_truth": target_window,
                            "performance": base_result,
                        }
                        quantile_levels = getattr(model, "quantile_levels", None)
                        if quantiles_result is not None and quantile_levels is not None:
                            prediction_record["quantile_levels"] = quantile_levels

                        predict_writer.append(prediction_record)
                finally:
                    predict_writer.close()
                    # Reset model state between users to avoid cross-user leakage.
                    # reset() is optional under the unified contract.
                    reset_fn = getattr(model, "reset", None)
                    if callable(reset_fn):
                        reset_fn()

                public_writer.increment_written(predict_writer.records_written)

        self._fallback_summary = self._summarize_fallback(
            model_name, fallback_substituted, fallback_total
        )

    @staticmethod
    def _apply_seasonal_naive_fallback(
        *,
        point_result: np.ndarray | None,
        history_window: np.ndarray,
        horizon: int,
        n_channels: int,
        fallback_model: SeasonalNaiveModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Substitute Seasonal-Naive predictions wherever the model emitted NaN.

        Returns the (possibly) repaired point forecast and the boolean mask of
        substituted positions, shape ``(n_channels, horizon)``.
        """
        if point_result is None:
            point_result = np.full((n_channels, horizon), np.nan, dtype=np.float32)
        fallback_mask = ~np.isfinite(point_result)
        if fallback_mask.any():
            fb_point, _fb_quantiles = fallback_model.predict(history_window, horizon)
            fb_point = np.asarray(fb_point, dtype=np.float32)
            point_result = point_result.copy()
            point_result[fallback_mask] = fb_point[fallback_mask]
        return point_result, fallback_mask

    def _summarize_fallback(
        self,
        model_name: str,
        substituted: np.ndarray | None,
        total: np.ndarray | None,
    ) -> dict:
        """Aggregate per-channel Seasonal-Naive fallback rates and warn if used."""
        if substituted is None or total is None or int(total.sum()) == 0:
            return {"overall_fallback_rate": 0.0, "fallback_rate": {}}

        total_cells = int(total.sum())
        subst_cells = int(substituted.sum())
        overall = subst_cells / total_cells
        per_channel = {
            f"ch_{i}": float(substituted[i] / total[i])
            for i in range(len(total))
            if int(total[i]) > 0
        }
        if subst_cells > 0:
            logger.warning(
                "[%s] Seasonal-Naive fallback substituted %.2f%% of forecast cells "
                "(%d / %d) where the model returned NaN. Per-channel rates: %s",
                model_name,
                100.0 * overall,
                subst_cells,
                total_cells,
                {k: round(v, 4) for k, v in per_channel.items() if v > 0.0},
            )
        return {"overall_fallback_rate": float(overall), "fallback_rate": per_channel}

    def _resolve_runtime_daily_start_hour_offset(self, model: BasePredictionModel) -> int:
        """Resolve the effective runtime offset for one evaluation run.

        The offset is a data-alignment concern (it changes which hours a window
        covers), not a model-capability drop. Models trained at a fixed offset
        expose ``training_daily_start_hour_offset``; the evaluator uses it and
        only errors when the eval config explicitly disagrees.
        """
        eval_offset = int(self.config.forecasting.daily_start_hour_offset)
        eval_offset_explicit = bool(
            getattr(self.config.forecasting, "_daily_start_hour_offset_explicit", False)
        )

        training_offset = getattr(model, "training_daily_start_hour_offset", None)
        if training_offset is None:
            return eval_offset

        training_offset = int(training_offset)
        if eval_offset_explicit and eval_offset != training_offset:
            raise ValueError(
                "Eval config forecasting.daily_start_hour_offset="
                f"{eval_offset} does not match checkpoint training offset "
                f"{training_offset}."
            )
        return training_offset
