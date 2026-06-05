"""Structured offline metrics pipeline for forecasting evaluation.

This module centralizes the offline forecasting-metrics workflow into one explicit
pipeline:

1. Load run config and dataset context
2. Read saved prediction parquet rows
3. Compute benchmark-global scales
4. Compute minimal-unit metrics for every ``(channel, hour-of-day)`` occurrence
5. Save metrics parquet outputs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from forecasting_evaluation.evaluation.point_metrics import (
    finalize_global_scales,
    finalize_hour_of_day_scales,
)
from forecasting_evaluation.metrics.offline.channel_merge import (
    merge_channel_first_array,
    resolve_channel_merges,
    zero_out_metrics_output,
)
from forecasting_evaluation.metrics.offline.common import (
    coerce_1d_float_array,
    coerce_2d_float_array,
    coerce_3d_float_array,
    coerce_non_negative_int,
    get_model_name,
    sanitize_name,
)
from forecasting_evaluation.metrics.offline.config_io import copy_run_config, load_run_config
from forecasting_evaluation.metrics.offline.data_context import (
    iter_offline_user_contexts_from_eval_flow,
)
from forecasting_evaluation.metrics.offline.metric_core import (
    MetricsComputer,
    slice_ground_truth,
)
from forecasting_evaluation.metrics.offline.parquet_io import (
    index_prediction_files,
    read_parquet_rows,
    save_metrics_result_by_metric,
)

logger = logging.getLogger(__name__)


class OfflineMetricsPipeline:
    """Explicit end-to-end offline metrics pipeline for one forecasting run."""

    def __init__(
        self,
        *,
        run_key: str,
        run_path: Path,
        metrics_output_path: Path,
        max_user: int | None = None,
        combine_channels: bool = True,
    ):
        """Initialize the offline metrics pipeline for one run."""
        self.run_key = str(run_key)
        self.run_path = Path(run_path)
        self.metrics_output_path = Path(metrics_output_path)
        self.max_user = int(max_user) if max_user is not None else None
        self.combine_channels = bool(combine_channels)

    def run(self) -> dict[str, Any]:
        """Execute the full structured metrics pipeline for one run."""
        pipeline_inputs = self.load_inputs()
        scale_result = self.compute_scales(pipeline_inputs)
        metrics_result = self.compute_minimal_unit_metrics(
            pipeline_inputs=pipeline_inputs,
            scale_result=scale_result,
        )
        save_result = self.save_metrics_result(
            pipeline_inputs=pipeline_inputs,
            metrics_result=metrics_result,
        )

        return {
            "run_key": self.run_key,
            "run_path": str(self.run_path),
            "model_name": pipeline_inputs["model_name"],
            "forecast_length": pipeline_inputs["forecast_length"],
            "scale_result": scale_result,
            "metrics_result": metrics_result,
            "save_result": save_result,
            "saved_rows": int(metrics_result["saved_rows"]),
            "skipped_rows": int(metrics_result["skipped_rows"]),
            "computed_user_count": int(metrics_result["computed_user_count"]),
            "max_user": self.max_user,
            "combine_channels": self.combine_channels,
            "output_run_dir": str(pipeline_inputs["output_run_dir"]),
            "metrics_dir": str(pipeline_inputs["metrics_dir"]),
        }

    def load_inputs(self) -> dict[str, Any]:
        """Step 1-2: load config, dataset context, and saved prediction parquet rows."""
        config = load_run_config(self.run_path)
        forecast_length = int(config.forecasting.forecasting_length)
        model_name = get_model_name(config.model)
        prediction_files = index_prediction_files(self.run_path, model_name)

        output_run_dir = self.metrics_output_path / sanitize_name(self.run_key)
        metrics_dir = output_run_dir

        return {
            "config": config,
            "forecast_length": forecast_length,
            "model_name": model_name,
            "output_run_dir": output_run_dir,
            "metrics_dir": metrics_dir,
            "prediction_files": prediction_files,
        }

    def iter_user_contexts(
        self,
        pipeline_inputs: dict[str, Any],
    ):
        """Yield merged offline user contexts without retaining every history."""
        for user_id, user_context in iter_offline_user_contexts_from_eval_flow(
            config=pipeline_inputs["config"],
            prediction_files=pipeline_inputs["prediction_files"],
        ):
            history = np.asarray(user_context["history"], dtype=float)
            variable_names = list(user_context["variable_names"])
            merge_plan = resolve_channel_merges(
                variable_names,
                combine_channels=self.combine_channels,
            )
            user_context["history"] = merge_channel_first_array(history, merge_plan)
            user_context["merge_plan"] = merge_plan
            yield user_id, user_context

    def compute_scales(self, pipeline_inputs: dict[str, Any]) -> dict[str, Any]:
        """Step 3: compute benchmark-global scales used by normalized metrics."""
        forecast_length = int(pipeline_inputs["forecast_length"])
        prediction_files = pipeline_inputs["prediction_files"]

        scale_sum: np.ndarray | None = None
        scale_count: np.ndarray | None = None
        global_scale_sum: np.ndarray | None = None
        global_scale_count: np.ndarray | None = None

        for user_id, user_context in self.iter_user_contexts(pipeline_inputs):
            history = np.asarray(user_context["history"], dtype=float)
            for parquet_path in prediction_files.get(user_id, []):
                rows = read_parquet_rows(parquet_path)
                if not rows:
                    continue

                history_lengths = [
                    history_length
                    for row in rows
                    if (history_length := coerce_non_negative_int(row.get("history_length"))) is not None
                ]
                (
                    scale_sum,
                    scale_count,
                    global_scale_sum,
                    global_scale_count,
                ) = self.accumulate_scale_statistics_for_starts(
                    values=history,
                    target_start_indices=np.asarray(history_lengths, dtype=int),
                    prediction_length=forecast_length,
                    season_length=24,
                    scale_sum=scale_sum,
                    scale_count=scale_count,
                    global_scale_sum=global_scale_sum,
                    global_scale_count=global_scale_count,
                )

                del rows

        global_scales_by_feature = None
        if scale_sum is not None and scale_count is not None:
            global_scales_by_feature = finalize_hour_of_day_scales(
                scale_sum=scale_sum,
                scale_count=scale_count,
            )
        global_scales_all_by_feature = None
        if global_scale_sum is not None and global_scale_count is not None:
            global_scales_all_by_feature = finalize_global_scales(
                scale_sum=global_scale_sum,
                scale_count=global_scale_count,
            )

        return {
            "season_length": 24,
            "global_scales_by_feature": global_scales_by_feature,
            "global_scales_all_by_feature": global_scales_all_by_feature,
        }

    @staticmethod
    def accumulate_scale_statistics_for_starts(
        *,
        values: np.ndarray,
        target_start_indices: np.ndarray,
        prediction_length: int,
        season_length: int,
        scale_sum: np.ndarray | None = None,
        scale_count: np.ndarray | None = None,
        global_scale_sum: np.ndarray | None = None,
        global_scale_count: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Accumulate hour-bucketed and global seasonal scales for many windows."""
        value_arr = np.asarray(values, dtype=float)
        if value_arr.ndim != 2:
            raise ValueError(f"Expected values with shape (n_features, trajectory_length), got {value_arr.shape}")
        if season_length <= 0:
            raise ValueError(f"season_length must be positive, got {season_length}")

        n_features, trajectory_length = value_arr.shape
        if scale_sum is None:
            scale_sum_arr = np.zeros((n_features, season_length), dtype=float)
        else:
            scale_sum_arr = np.asarray(scale_sum, dtype=float)
        if scale_count is None:
            scale_count_arr = np.zeros((n_features, season_length), dtype=np.int64)
        else:
            scale_count_arr = np.asarray(scale_count, dtype=np.int64)
        if global_scale_sum is None:
            global_scale_sum_arr = np.zeros(n_features, dtype=float)
        else:
            global_scale_sum_arr = np.asarray(global_scale_sum, dtype=float).reshape(-1)
        if global_scale_count is None:
            global_scale_count_arr = np.zeros(n_features, dtype=np.int64)
        else:
            global_scale_count_arr = np.asarray(global_scale_count, dtype=np.int64).reshape(-1)

        starts = np.asarray(target_start_indices, dtype=int).reshape(-1)
        if starts.size == 0 or prediction_length <= 0:
            return scale_sum_arr, scale_count_arr, global_scale_sum_arr, global_scale_count_arr

        offsets = np.arange(prediction_length, dtype=int)
        target_indices = starts[:, None] + offsets[None, :]
        previous_indices = target_indices - season_length
        valid_pair_mask = (
            (target_indices >= 0)
            & (target_indices < trajectory_length)
            & (previous_indices >= 0)
        )
        if not np.any(valid_pair_mask):
            return scale_sum_arr, scale_count_arr, global_scale_sum_arr, global_scale_count_arr

        target_flat = target_indices[valid_pair_mask]
        previous_flat = previous_indices[valid_pair_mask]
        hour_flat = target_flat % season_length

        current_values = value_arr[:, target_flat]
        previous_values = value_arr[:, previous_flat]
        valid_observed = np.isfinite(current_values) & np.isfinite(previous_values)
        if not np.any(valid_observed):
            return scale_sum_arr, scale_count_arr, global_scale_sum_arr, global_scale_count_arr

        absolute_diff = np.abs(current_values - previous_values)
        feature_idx, pair_idx = np.nonzero(valid_observed)
        diff_values = absolute_diff[feature_idx, pair_idx]
        hour_idx = hour_flat[pair_idx]

        np.add.at(scale_sum_arr, (feature_idx, hour_idx), diff_values)
        np.add.at(scale_count_arr, (feature_idx, hour_idx), 1)
        np.add.at(global_scale_sum_arr, feature_idx, diff_values)
        np.add.at(global_scale_count_arr, feature_idx, 1)

        return scale_sum_arr, scale_count_arr, global_scale_sum_arr, global_scale_count_arr

    def compute_minimal_unit_metrics(
        self,
        *,
        pipeline_inputs: dict[str, Any],
        scale_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Step 4: compute all minimal-unit metrics for every saved prediction row."""
        forecast_length = int(pipeline_inputs["forecast_length"])
        model_name = str(pipeline_inputs["model_name"])
        prediction_files = pipeline_inputs["prediction_files"]
        metrics_dir = Path(pipeline_inputs["metrics_dir"])

        metrics_computer = MetricsComputer(
            global_scales_by_feature=scale_result.get("global_scales_by_feature"),
            global_scales_all_by_feature=scale_result.get("global_scales_all_by_feature"),
            season_length=int(scale_result.get("season_length", 24)),
        )

        model_key = sanitize_name(model_name)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        saved_rows = 0
        skipped_rows = 0
        computed_user_count = 0
        saved_files_by_metric: dict[str, dict[str, str]] = {}

        for user_id, user_context in self.iter_user_contexts(pipeline_inputs):
            if self.max_user is not None and computed_user_count >= self.max_user:
                logger.info(
                    "Reached max_user limit for run=%s: max_user=%s",
                    self.run_key,
                    self.max_user,
                )
                break

            user_metrics_file = metrics_dir / "mae" / f"{sanitize_name(user_id)}.parquet"
            if user_metrics_file.exists():
                logger.info(
                    "Metrics parquet for user exists, skipping user. run=%s user=%s file=%s",
                    self.run_key,
                    user_id,
                    user_metrics_file,
                )
                continue

            history = np.asarray(user_context["history"], dtype=float)
            variable_names = list(user_context["variable_names"])
            merge_plan = user_context["merge_plan"]
            user_records: list[dict[str, Any]] = []

            for parquet_path in prediction_files.get(user_id, []):
                rows = read_parquet_rows(parquet_path)
                if not rows:
                    continue

                for row in rows:
                    history_length = coerce_non_negative_int(row.get("history_length"))
                    if history_length is None:
                        skipped_rows += 1
                        continue

                    gt_pack = slice_ground_truth(
                        history=history,
                        history_length=history_length,
                        forecast_length=forecast_length,
                    )
                    if gt_pack is None:
                        skipped_rows += 1
                        continue

                    ground_truth, ground_truth_observed_mask = gt_pack
                    point_predictions = merge_channel_first_array(
                        coerce_2d_float_array(row.get("point_predictions")),
                        merge_plan,
                    )
                    quantile_predictions = merge_channel_first_array(
                        coerce_3d_float_array(row.get("quantile_predictions")),
                        merge_plan,
                    )
                    metrics_output = metrics_computer.compute(
                        point_predictions=point_predictions,
                        quantile_predictions=quantile_predictions,
                        full_trajectory=history,
                        target_start_idx=history_length,
                        ground_truth=ground_truth,
                        ground_truth_observed_mask=ground_truth_observed_mask,
                        variable_names=variable_names,
                        quantile_levels=coerce_1d_float_array(row.get("quantile_levels")),
                    )
                    metrics_output = zero_out_metrics_output(
                        metrics_output,
                        merge_plan.zero_feature_indices,
                    )

                    record = {
                        "user_id": user_id,
                        "model": model_name,
                        "history_length": history_length,
                        "forecasting_length": forecast_length,
                        "mae": metrics_output.get("mae"),
                        "mse": metrics_output.get("mse"),
                        "mase": metrics_output.get("mase"),
                        "mase_all": metrics_output.get("mase_all"),
                        "ql": metrics_output.get("ql"),
                        "sql": metrics_output.get("sql"),
                    }
                    perf = row.get("performance") or {}
                    record.update({f"perf_{k}": v for k, v in perf.items()})
                    user_records.append(record)
                    saved_rows += 1

                del rows

            if user_records:
                user_saved_files_by_metric = save_metrics_result_by_metric(
                    output_root=metrics_dir,
                    model_key=model_key,
                    records_by_user={user_id: user_records},
                )
                for metric_name, saved_files in user_saved_files_by_metric.items():
                    saved_files_by_metric.setdefault(metric_name, {}).update(saved_files)
                computed_user_count += 1

        return {
            "saved_files_by_metric": saved_files_by_metric,
            "saved_rows": saved_rows,
            "skipped_rows": skipped_rows,
            "computed_user_count": computed_user_count,
        }

    def save_metrics_result(
        self,
        *,
        pipeline_inputs: dict[str, Any],
        metrics_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Step 5: persist computed metrics parquet outputs to disk."""
        output_run_dir = Path(pipeline_inputs["output_run_dir"])
        metrics_dir = Path(pipeline_inputs["metrics_dir"])
        model_key = sanitize_name(str(pipeline_inputs["model_name"]))

        output_run_dir.mkdir(parents=True, exist_ok=True)
        copy_run_config(self.run_path, output_run_dir)

        return {
            "saved_files_by_metric": metrics_result["saved_files_by_metric"],
            "output_run_dir": str(output_run_dir),
            "metrics_dir": str(metrics_dir),
        }
