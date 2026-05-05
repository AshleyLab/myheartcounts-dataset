"""Metric core and ground-truth slicing for offline forecasting metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from forecasting_evaluation.evaluation.point_metrics import compute_mase, compute_mase_all
from forecasting_evaluation.evaluation.quantiles_metrics import compute_quantiles_metrics
from forecasting_evaluation.metrics.offline.common import resolve_quantile_levels


def to_2d_float_list(values: np.ndarray) -> list[list[float]]:
    """Convert 2D array to nested float lists, preserving NaN for missing values."""
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {values.shape}")

    result: list[list[float]] = []
    for row in values:
        row_list: list[float] = []
        for x in row:
            val = float(x)
            row_list.append(val if np.isfinite(val) else float("nan"))
        result.append(row_list)
    return result


def slice_ground_truth(
    history: np.ndarray,
    history_length: int,
    forecast_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Slice GT segment and derive observed mask from finite values."""
    if history.ndim != 2:
        return None

    start = history_length
    end = start + forecast_length
    if start < 0 or end > history.shape[1]:
        return None

    ground_truth = history[:, start:end]
    ground_truth_observed_mask = np.isfinite(ground_truth)
    return ground_truth, ground_truth_observed_mask


def ensure_quantile_predictions(
    *,
    point_predictions: np.ndarray | None,
    quantile_predictions: np.ndarray | None,
) -> np.ndarray | None:
    """Fallback missing quantile forecasts to a single point-based quantile.

    When a model only provides point forecasts, offline QL/sQL still expects a
    quantile tensor. In that case, wrap the point forecast as a single quantile
    level so downstream code can resolve it to the default median quantile.
    """
    if quantile_predictions is not None:
        return quantile_predictions

    if point_predictions is None or point_predictions.ndim != 2:
        return None

    return point_predictions[:, :, np.newaxis]


class MetricsComputer:
    """Compute per-sample forecasting metrics matrices by feature and horizon."""

    def __init__(
        self,
        global_scales_by_feature: np.ndarray | None = None,
        global_scales_all_by_feature: np.ndarray | None = None,
        season_length: int = 24,
    ):
        """Initialize metric scaling inputs and season length."""
        self.global_scales_by_feature = (
            None if global_scales_by_feature is None else np.asarray(global_scales_by_feature, dtype=float)
        )
        self.global_scales_all_by_feature = (
            None if global_scales_all_by_feature is None else np.asarray(global_scales_all_by_feature, dtype=float)
        )
        self.season_length = int(season_length)

    def compute(
        self,
        *,
        point_predictions: np.ndarray | None,
        quantile_predictions: np.ndarray | None,
        full_trajectory: np.ndarray,
        target_start_idx: int,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
        variable_names: list[str],
        quantile_levels: np.ndarray | None,
    ) -> dict[str, Any]:
        """Compute offline metrics for one prediction sample.

        Args:
            point_predictions: Point forecast matrix, shape (n_features, horizon), or None.
            quantile_predictions: Quantile forecast tensor, shape
                (n_features, horizon, n_quantiles), or None.
            full_trajectory: Full transformed trajectory for the user, shape
                (n_features, trajectory_length), used for MASE/sQL scaling.
            target_start_idx: Absolute 0-based start index of ``ground_truth`` inside
                ``full_trajectory``.
            ground_truth: Ground-truth matrix, shape (n_features, horizon).
            ground_truth_observed_mask: Boolean observed-value mask with same shape as
                ground_truth.
            variable_names: Feature names aligned with first dimension.
            quantile_levels: Quantile levels aligned with quantile_predictions, if present.

        Returns:
            Dict containing flattened metric matrices under keys ``mae``, ``mse``,
            ``mase``, ``mase_all``, ``ql``, and ``sql``.
        """
        quantile_predictions = ensure_quantile_predictions(
            point_predictions=point_predictions,
            quantile_predictions=quantile_predictions,
        )

        mae_result = self._compute_mae(
            point_predictions=point_predictions,
            ground_truth=ground_truth,
            ground_truth_observed_mask=ground_truth_observed_mask,
        )
        mse_result = self._compute_mse(
            point_predictions=point_predictions,
            ground_truth=ground_truth,
            ground_truth_observed_mask=ground_truth_observed_mask,
        )
        ql_result, sql_result = self._compute_quantile_metrics(
            quantile_predictions=quantile_predictions,
            full_trajectory=full_trajectory,
            target_start_idx=target_start_idx,
            ground_truth=ground_truth,
            ground_truth_observed_mask=ground_truth_observed_mask,
            variable_names=variable_names,
            quantile_levels=quantile_levels,
        )
        mase_result = self._compute_mase(
            point_predictions=point_predictions,
            full_trajectory=full_trajectory,
            target_start_idx=target_start_idx,
            ground_truth=ground_truth,
            ground_truth_observed_mask=ground_truth_observed_mask,
        )
        mase_all_result = self._compute_mase_all(
            point_predictions=point_predictions,
            ground_truth=ground_truth,
            ground_truth_observed_mask=ground_truth_observed_mask,
        )
        return {
            "mae": mae_result,
            "mse": mse_result,
            "mase": mase_result,
            "mase_all": mase_all_result,
            "ql": ql_result,
            "sql": sql_result,
        }

    def _compute_mae(
        self,
        *,
        point_predictions: np.ndarray | None,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
    ) -> list[list[float]]:
        if point_predictions is None or point_predictions.shape != ground_truth.shape:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        abs_error = np.abs(point_predictions - ground_truth)
        masked_error = np.where(ground_truth_observed_mask, abs_error, np.nan)
        return to_2d_float_list(masked_error)

    def _compute_mse(
        self,
        *,
        point_predictions: np.ndarray | None,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
    ) -> list[list[float]]:
        if point_predictions is None or point_predictions.shape != ground_truth.shape:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        squared_error = (point_predictions - ground_truth) ** 2
        masked_error = np.where(ground_truth_observed_mask, squared_error, np.nan)
        return to_2d_float_list(masked_error)

    def _compute_quantile_metrics(
        self,
        *,
        quantile_predictions: np.ndarray | None,
        full_trajectory: np.ndarray,
        target_start_idx: int,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
        variable_names: list[str],
        quantile_levels: np.ndarray | None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        if quantile_predictions is None or quantile_predictions.ndim != 3:
            empty = to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))
            return empty, empty

        if quantile_predictions.shape[:2] != ground_truth.shape:
            empty = to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))
            return empty, empty

        n_quantiles = quantile_predictions.shape[2]
        resolved_levels = resolve_quantile_levels(quantile_levels, n_quantiles)

        quantiles_metrics = compute_quantiles_metrics(
            predictions=quantile_predictions,
            ground_truth=ground_truth,
            variable_names=variable_names,
            quantile_levels=resolved_levels,
            scales_by_feature=self.global_scales_by_feature,
            mask=ground_truth_observed_mask,
            target_start_idx=target_start_idx,
        )

        ql_by_feature = quantiles_metrics.get("ql")
        sql_by_feature = quantiles_metrics.get("sQL")
        if not isinstance(ql_by_feature, dict) or not isinstance(sql_by_feature, dict):
            empty = to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))
            return empty, empty

        ql_matrix = np.full(ground_truth.shape, np.nan, dtype=float)
        sql_matrix = np.full(ground_truth.shape, np.nan, dtype=float)
        for feature_idx, var_name in enumerate(variable_names):
            ql_values = ql_by_feature.get(var_name)
            sql_values = sql_by_feature.get(var_name)
            if ql_values is None:
                continue
            ql_values_arr = np.asarray(ql_values, dtype=float)
            if ql_values_arr.ndim != 1 or ql_values_arr.shape[0] != ground_truth.shape[1]:
                continue
            ql_matrix[feature_idx] = ql_values_arr

            if sql_values is None:
                continue
            sql_values_arr = np.asarray(sql_values, dtype=float)
            if sql_values_arr.ndim != 1 or sql_values_arr.shape[0] != ground_truth.shape[1]:
                continue
            sql_matrix[feature_idx] = sql_values_arr

        return to_2d_float_list(ql_matrix), to_2d_float_list(sql_matrix)

    def _compute_mase(
        self,
        *,
        point_predictions: np.ndarray | None,
        full_trajectory: np.ndarray,
        target_start_idx: int,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
    ) -> list[list[float]]:
        if point_predictions is None or point_predictions.shape != ground_truth.shape:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        if self.global_scales_by_feature is None:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        if self.global_scales_by_feature.ndim != 2:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        if self.global_scales_by_feature.shape[0] != ground_truth.shape[0]:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        mase_matrix = np.full(ground_truth.shape, np.nan, dtype=float)
        for feature_idx in range(ground_truth.shape[0]):
            feature_mase = np.asarray(
                compute_mase(
                    predictions=point_predictions[feature_idx : feature_idx + 1, :],
                    ground_truth=ground_truth[feature_idx : feature_idx + 1, :],
                    scales_by_hour=self.global_scales_by_feature[feature_idx],
                    season_length=self.season_length,
                    target_start_idx=target_start_idx,
                ),
                dtype=float,
            )
            mase_matrix[feature_idx] = np.where(
                ground_truth_observed_mask[feature_idx],
                feature_mase,
                np.nan,
            )

        return to_2d_float_list(mase_matrix)

    def _compute_mase_all(
        self,
        *,
        point_predictions: np.ndarray | None,
        ground_truth: np.ndarray,
        ground_truth_observed_mask: np.ndarray,
    ) -> list[list[float]]:
        if point_predictions is None or point_predictions.shape != ground_truth.shape:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        if self.global_scales_all_by_feature is None:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        global_scales = np.asarray(self.global_scales_all_by_feature, dtype=float).reshape(-1)
        if global_scales.shape[0] != ground_truth.shape[0]:
            return to_2d_float_list(np.full(ground_truth.shape, np.nan, dtype=float))

        mase_all_matrix = np.full(ground_truth.shape, np.nan, dtype=float)
        for feature_idx in range(ground_truth.shape[0]):
            feature_mase_all = np.asarray(
                compute_mase_all(
                    predictions=point_predictions[feature_idx : feature_idx + 1, :],
                    ground_truth=ground_truth[feature_idx : feature_idx + 1, :],
                    global_scale=float(global_scales[feature_idx]),
                ),
                dtype=float,
            )
            mase_all_matrix[feature_idx] = np.where(
                ground_truth_observed_mask[feature_idx],
                feature_mase_all,
                np.nan,
            )

        return to_2d_float_list(mase_all_matrix)
