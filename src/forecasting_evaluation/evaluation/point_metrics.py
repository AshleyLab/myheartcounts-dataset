"""Point forecasting metrics for forecasting evaluation."""

from __future__ import annotations

import numpy as np


def compute_mae(predictions: np.ndarray, ground_truth: np.ndarray) -> list[float]:
    r"""Compute per-horizon Mean Absolute Error.

    Formula (per horizon $t$):
        $\mathrm{MAE}_t = |y_t - \hat{y}_t|$

    Args:
        predictions: Predicted values with shape ``(1, prediction_length)``.
        ground_truth: Ground-truth values with shape ``(1, prediction_length)``.

    Returns:
        A flat list of length ``prediction_length`` containing the absolute error at
        each forecast horizon.
    """
    return np.abs(predictions - ground_truth).reshape(-1).tolist()


def compute_mse(predictions: np.ndarray, ground_truth: np.ndarray) -> list[float]:
    r"""Compute per-horizon Mean Squared Error.

    Formula (per horizon $t$):
        $\mathrm{MSE}_t = (y_t - \hat{y}_t)^2$

    Args:
        predictions: Predicted values with shape ``(1, prediction_length)``.
        ground_truth: Ground-truth values with shape ``(1, prediction_length)``.

    Returns:
        A flat list of length ``prediction_length`` containing the squared error at
        each forecast horizon.
    """
    return ((predictions - ground_truth) ** 2).reshape(-1).tolist()


def compute_f1(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    r"""Compute F1 score over all valid positions.

    Args:
        predictions: Predicted probabilities or scores. The array can have any shape
            as long as it is broadcast-compatible with ``ground_truth`` after
            flattening.
        ground_truth: Binary ground-truth values. Entries equal to ``1.0`` are treated
            as positive, and all other finite values are treated as negative.
        mask: Optional boolean mask with the same flattened shape as
            ``ground_truth``. Positions where the mask is ``False`` are excluded from
            the TP/FP/FN counts.
        threshold: Threshold used to convert prediction scores into binary labels.

    Returns:
        Scalar F1 score computed from the aggregated TP/FP/FN counts over all valid
        positions. Returns ``NaN`` when ``2 * TP + FP + FN == 0``.
    """
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    truth = np.asarray(ground_truth, dtype=float).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(
            f"Predictions and ground truth must have the same flattened shape, got {pred.shape} and {truth.shape}"
        )

    if mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool).reshape(-1)
        if valid.shape != truth.shape:
            raise ValueError(
                f"Mask must have the same flattened shape as ground truth, got {valid.shape} and {truth.shape}"
            )

    pred_binary = pred >= float(threshold)
    truth_binary = truth == 1.0

    tp = int(np.sum(valid & truth_binary & pred_binary))
    fp = int(np.sum(valid & (~truth_binary) & pred_binary))
    fn = int(np.sum(valid & truth_binary & (~pred_binary)))

    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return float("nan")
    return float((2.0 * tp) / denominator)


def finalize_hour_of_day_scales(
    scale_sum: np.ndarray,
    scale_count: np.ndarray,
) -> np.ndarray:
    """Convert accumulated scale sums/counts into mean hour-of-day scales.

    This helper is the final reduction step for the benchmark-global scaling term
    used by ``MASE`` and ``sQL``. It divides the accumulated absolute-difference sum
    by the number of valid seasonal pairs for each ``(feature, hour-of-day)`` cell.

    Args:
        scale_sum: Sum of absolute seasonal differences. Shape is typically
            ``(n_features, season_length)``.
        scale_count: Count of valid seasonal pairs contributing to each cell. Must
            have the same shape as ``scale_sum``.

    Returns:
        An array with the same shape as the inputs. Cells with zero valid pairs, or
        cells whose resulting mean is ``0.0``, are returned as ``NaN`` so downstream
        normalized metrics remain undefined in those cases.
    """
    sum_arr = np.asarray(scale_sum, dtype=float)
    count_arr = np.asarray(scale_count, dtype=float)
    if sum_arr.shape != count_arr.shape:
        raise ValueError(
            f"scale_sum and scale_count must have the same shape, got {sum_arr.shape} and {count_arr.shape}"
        )

    scales = np.full(sum_arr.shape, np.nan, dtype=float)
    valid = count_arr > 0
    scales[valid] = sum_arr[valid] / count_arr[valid]
    return np.where(np.isfinite(scales) & (scales != 0.0), scales, np.nan)


def finalize_global_scales(
    scale_sum: np.ndarray,
    scale_count: np.ndarray,
) -> np.ndarray:
    """Convert accumulated scale sums/counts into one global scale per feature.

    This helper is used for ``mase_all``, whose denominator is a single
    benchmark-global seasonal-naive scale per feature instead of one scale per
    hour-of-day bucket.

    Args:
        scale_sum: Sum of absolute seasonal differences with shape ``(n_features,)``.
        scale_count: Count of valid seasonal pairs with the same shape.

    Returns:
        One-dimensional array of length ``n_features``. Entries are ``NaN`` when a
        feature has no valid seasonal pairs or when the resulting scale is ``0.0``.
    """
    sum_arr = np.asarray(scale_sum, dtype=float).reshape(-1)
    count_arr = np.asarray(scale_count, dtype=float).reshape(-1)
    if sum_arr.shape != count_arr.shape:
        raise ValueError(
            f"scale_sum and scale_count must have the same shape, got {sum_arr.shape} and {count_arr.shape}"
        )

    scales = np.full(sum_arr.shape, np.nan, dtype=float)
    valid = count_arr > 0
    scales[valid] = sum_arr[valid] / count_arr[valid]
    return np.where(np.isfinite(scales) & (scales != 0.0), scales, np.nan)


def accumulate_hour_of_day_scale_statistics(
    values: np.ndarray,
    target_start_idx: int,
    prediction_length: int,
    season_length: int,
    scale_sum: np.ndarray | None = None,
    scale_count: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Accumulate benchmark-level scale statistics for one forecasting window.

    This follows the manuscript definition

    .. math::
        S_h^{(c)} \propto
        \sum_{t \in \mathcal{T}_i}
        M_{i,t+h}^{(c)} M_{i,t+h-24}^{(c)}
        \lvert Y_{i,t+h}^{(c)} - Y_{i,t+h-24}^{(c)} \rvert.

    Args:
        values: Full transformed trajectory for one user with shape
            ``(n_features, trajectory_length)``. Missing target values should already
            be encoded as ``NaN`` so they can be excluded via the implicit mask.
        target_start_idx: Absolute 0-based start index of the forecasting window
            inside ``values``.
        prediction_length: Forecast horizon ``H``. The function examines target
            indices ``target_start_idx`` through ``target_start_idx + H - 1``.
        season_length: Seasonal lag in hours. For the benchmark this is ``24``.
        scale_sum: Optional running sum array used to accumulate statistics across
            many forecasting windows. When omitted, a zero-initialized array is
            created.
        scale_count: Optional running count array with the same shape as
            ``scale_sum``. When omitted, a zero-initialized array is created.

    Returns:
        A tuple ``(scale_sum, scale_count)`` after incorporating this forecasting
        window's valid seasonal pairs. Only pairs where both ``t+h`` and
        ``t+h-24`` are finite contribute to the accumulation.
    """
    value_arr = np.asarray(values, dtype=float)
    if value_arr.ndim != 2:
        raise ValueError(
            f"Expected values with shape (n_features, trajectory_length), got {value_arr.shape}"
        )
    if season_length <= 0:
        raise ValueError(f"season_length must be positive, got {season_length}")

    n_features, trajectory_length = value_arr.shape
    if scale_sum is None:
        scale_sum_arr = np.zeros((n_features, season_length), dtype=float)
    else:
        scale_sum_arr = np.asarray(scale_sum, dtype=float)
        if scale_sum_arr.shape != (n_features, season_length):
            raise ValueError(
                f"scale_sum must have shape {(n_features, season_length)}, got {scale_sum_arr.shape}"
            )

    if scale_count is None:
        scale_count_arr = np.zeros((n_features, season_length), dtype=np.int64)
    else:
        scale_count_arr = np.asarray(scale_count, dtype=np.int64)
        if scale_count_arr.shape != (n_features, season_length):
            raise ValueError(
                f"scale_count must have shape {(n_features, season_length)}, got {scale_count_arr.shape}"
            )

    if prediction_length <= 0:
        return scale_sum_arr, scale_count_arr

    target_indices = np.arange(target_start_idx, target_start_idx + prediction_length, dtype=int)
    valid_target_mask = (target_indices >= 0) & (target_indices < trajectory_length)
    if not np.any(valid_target_mask):
        return scale_sum_arr, scale_count_arr

    target_indices = target_indices[valid_target_mask]
    previous_indices = target_indices - season_length
    valid_pair_mask = previous_indices >= 0
    if not np.any(valid_pair_mask):
        return scale_sum_arr, scale_count_arr

    target_indices = target_indices[valid_pair_mask]
    previous_indices = previous_indices[valid_pair_mask]
    hour_buckets = target_indices % season_length

    current_values = value_arr[:, target_indices]
    previous_values = value_arr[:, previous_indices]
    valid_observed = np.isfinite(current_values) & np.isfinite(previous_values)
    absolute_diff = np.abs(current_values - previous_values)

    for col_idx, hour_bucket in enumerate(hour_buckets):
        feature_mask = valid_observed[:, col_idx]
        if not np.any(feature_mask):
            continue
        scale_sum_arr[feature_mask, hour_bucket] += absolute_diff[feature_mask, col_idx]
        scale_count_arr[feature_mask, hour_bucket] += 1

    return scale_sum_arr, scale_count_arr


def accumulate_global_scale_statistics(
    values: np.ndarray,
    target_start_idx: int,
    prediction_length: int,
    season_length: int,
    scale_sum: np.ndarray | None = None,
    scale_count: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Accumulate one benchmark-global seasonal scale per feature.

    Unlike ``accumulate_hour_of_day_scale_statistics()``, this helper ignores the
    hour-of-day bucket and aggregates every valid seasonal pair for a feature into a
    single global denominator. This matches the ``mase_all`` definition requested by
    the benchmark.

    Args:
        values: Full transformed trajectory for one user with shape
            ``(n_features, trajectory_length)``.
        target_start_idx: Absolute 0-based start index of the forecasting window.
        prediction_length: Forecast horizon ``H``.
        season_length: Seasonal lag in hours. For the benchmark this is ``24``.
        scale_sum: Optional running per-feature sum of absolute seasonal
            differences.
        scale_count: Optional running per-feature count of valid seasonal pairs.

    Returns:
        A tuple ``(scale_sum, scale_count)`` after incorporating this forecasting
        window's valid seasonal pairs into one global scale per feature.
    """
    value_arr = np.asarray(values, dtype=float)
    if value_arr.ndim != 2:
        raise ValueError(
            f"Expected values with shape (n_features, trajectory_length), got {value_arr.shape}"
        )
    if season_length <= 0:
        raise ValueError(f"season_length must be positive, got {season_length}")

    n_features, trajectory_length = value_arr.shape
    if scale_sum is None:
        scale_sum_arr = np.zeros(n_features, dtype=float)
    else:
        scale_sum_arr = np.asarray(scale_sum, dtype=float).reshape(-1)
        if scale_sum_arr.shape != (n_features,):
            raise ValueError(
                f"scale_sum must have shape {(n_features,)}, got {scale_sum_arr.shape}"
            )

    if scale_count is None:
        scale_count_arr = np.zeros(n_features, dtype=np.int64)
    else:
        scale_count_arr = np.asarray(scale_count, dtype=np.int64).reshape(-1)
        if scale_count_arr.shape != (n_features,):
            raise ValueError(
                f"scale_count must have shape {(n_features,)}, got {scale_count_arr.shape}"
            )

    if prediction_length <= 0:
        return scale_sum_arr, scale_count_arr

    target_indices = np.arange(target_start_idx, target_start_idx + prediction_length, dtype=int)
    valid_target_mask = (target_indices >= 0) & (target_indices < trajectory_length)
    if not np.any(valid_target_mask):
        return scale_sum_arr, scale_count_arr

    target_indices = target_indices[valid_target_mask]
    previous_indices = target_indices - season_length
    valid_pair_mask = previous_indices >= 0
    if not np.any(valid_pair_mask):
        return scale_sum_arr, scale_count_arr

    target_indices = target_indices[valid_pair_mask]
    previous_indices = previous_indices[valid_pair_mask]

    current_values = value_arr[:, target_indices]
    previous_values = value_arr[:, previous_indices]
    valid_observed = np.isfinite(current_values) & np.isfinite(previous_values)
    absolute_diff = np.abs(current_values - previous_values)

    feature_sums = np.where(valid_observed, absolute_diff, 0.0).sum(axis=1)
    feature_counts = valid_observed.sum(axis=1)
    scale_sum_arr += feature_sums
    scale_count_arr += feature_counts
    return scale_sum_arr, scale_count_arr


def select_hour_of_day_scale(
    scales_by_hour: np.ndarray,
    season_length: int,
    target_idx: int,
) -> float:
    """Select the relevant hour-of-day scale for one absolute target time.

    Args:
        scales_by_hour: One-dimensional scale vector of length ``season_length``.
        season_length: Seasonal lag in hours.
        target_idx: Absolute 0-based target time index.

    Returns:
        The scale associated with the hour-of-day bucket of ``target_idx``. Returns
        ``NaN`` when ``target_idx`` is negative.
    """
    if target_idx < 0:
        return float("nan")

    scales = np.asarray(scales_by_hour, dtype=float).reshape(-1)
    if scales.shape[0] != season_length:
        raise ValueError(f"scales_by_hour must have length {season_length}, got {scales.shape[0]}")
    return float(scales[int(target_idx % season_length)])


def compute_mase(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    scales_by_hour: np.ndarray,
    season_length: int,
    target_start_idx: int,
) -> list[float]:
    r"""Compute per-horizon Mean Absolute Scaled Error using benchmark-global scales.

    Formula (per horizon $t+h$):
        $\mathrm{MASE}_{t+h} = \frac{|y_{t+h} - \hat{y}_{t+h}|}{S^{(H_{t+h})}}$

    Here ``S^(r)`` is the benchmark-global scaling term for hour-of-day bucket
    ``r``, precomputed from all valid evaluation-window seasonal pairs.

    Args:
        predictions: Predicted values with shape ``(1, prediction_length)``.
        ground_truth: Ground-truth values with shape ``(1, prediction_length)``.
        scales_by_hour: One-dimensional scale vector of length ``season_length`` for
            the current feature.
        season_length: Seasonal lag in hours. For the benchmark this is ``24``.
        target_start_idx: Absolute 0-based start index of the forecast window inside
            the full trajectory. This is used only to map each horizon to the correct
            hour-of-day bucket.

    Returns:
        A flat list of length ``prediction_length``. Each entry is the absolute error
        divided by the corresponding hour-of-day scale. If the required scale is
        missing or zero, the returned value is ``NaN``.
    """
    pred_errors = np.abs(predictions - ground_truth).reshape(-1)
    scales = np.asarray(scales_by_hour, dtype=float).reshape(-1)
    if scales.shape[0] != season_length:
        raise ValueError(f"scales_by_hour must have length {season_length}, got {scales.shape[0]}")

    target_indices = np.arange(target_start_idx, target_start_idx + pred_errors.shape[0], dtype=int)
    scale_by_horizon = np.full(pred_errors.shape, np.nan, dtype=float)
    valid_index_mask = target_indices >= 0
    scale_by_horizon[valid_index_mask] = scales[target_indices[valid_index_mask] % season_length]

    with np.errstate(divide="ignore", invalid="ignore"):
        mase = pred_errors / scale_by_horizon
    mase = np.where(np.isfinite(scale_by_horizon) & (scale_by_horizon != 0.0), mase, np.nan)
    return mase.tolist()


def compute_mase_all(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    global_scale: float,
) -> list[float]:
    r"""Compute per-horizon Mean Absolute Scaled Error with one global feature scale.

    Formula (per horizon $t+h$):
        $\mathrm{MASE\_all}_{t+h} = \frac{|y_{t+h} - \hat{y}_{t+h}|}{S_{\mathrm{all}}}$

    Args:
        predictions: Predicted values with shape ``(1, prediction_length)``.
        ground_truth: Ground-truth values with shape ``(1, prediction_length)``.
        global_scale: Single benchmark-global seasonal scale for the current feature.

    Returns:
        A flat list of length ``prediction_length``. Each entry is the absolute error
        divided by ``global_scale``. Returns ``NaN`` for all horizons when the scale
        is missing or zero.
    """
    pred_errors = np.abs(predictions - ground_truth).reshape(-1)
    if not np.isfinite(global_scale) or float(global_scale) == 0.0:
        return np.full(pred_errors.shape, np.nan, dtype=float).tolist()
    return (pred_errors / float(global_scale)).tolist()


def compute_point_metrics(
    predictions: np.ndarray | None,
    ground_truth: np.ndarray,
    variable_names: list[str] | None = None,
    scales_by_feature: np.ndarray | None = None,
    season_length: int = 24,
    target_start_idx: int | None = None,
) -> dict[str, dict[str, list[float]] | None]:
    """Compute point forecasting metrics for all features in one sample.

    Args:
        predictions: Point predictions with shape ``(n_features, prediction_length)``.
            When ``None``, all metric groups are returned as ``None``.
        ground_truth: Ground-truth matrix with the same shape as ``predictions``.
        variable_names: Optional feature names aligned with the first dimension. If
            omitted, default names ``var_0``, ``var_1``, ... are generated.
        scales_by_feature: Optional benchmark-global scale matrix with shape
            ``(n_features, season_length)``. When provided together with
            ``target_start_idx``, ``MASE`` is computed per feature.
        season_length: Seasonal lag in hours used by ``MASE``.
        target_start_idx: Absolute 0-based forecast start index used to map each
            horizon to its hour-of-day bucket.

    Returns:
        A dictionary with the following keys:
        - ``mae``: maps each variable name to its per-horizon MAE list.
        - ``mse``: maps each variable name to its per-horizon MSE list.
        - ``f1``: maps each variable name to a scalar F1 value computed over the full
          horizon.
        - ``mase``: maps each variable name to its per-horizon MASE list when scales
          are provided; otherwise ``None``.
    """
    if predictions is None:
        return {
            "mae": None,
            "mse": None,
            "f1": None,
            "mase": None,
        }

    n_features = predictions.shape[0]
    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(n_features)]

    mae_per_feature: dict[str, list[float]] = {}
    mse_per_feature: dict[str, list[float]] = {}
    f1_per_feature: dict[str, float] = {}
    mase_per_feature: dict[str, list[float]] = {}

    for i, var_name in enumerate(variable_names):
        feature_pred = predictions[i : i + 1, :]
        feature_truth = ground_truth[i : i + 1, :]

        mae_per_feature[var_name] = compute_mae(feature_pred, feature_truth)
        mse_per_feature[var_name] = compute_mse(feature_pred, feature_truth)
        f1_per_feature[var_name] = compute_f1(feature_pred, feature_truth)

        if scales_by_feature is not None and target_start_idx is not None:
            mase_per_feature[var_name] = compute_mase(
                predictions=feature_pred,
                ground_truth=feature_truth,
                scales_by_hour=scales_by_feature[i],
                season_length=season_length,
                target_start_idx=target_start_idx,
            )

    return {
        "mae": mae_per_feature,
        "mse": mse_per_feature,
        "f1": f1_per_feature,
        "mase": mase_per_feature if mase_per_feature else None,
    }
