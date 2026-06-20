"""Quantile forecasting metrics for forecasting evaluation."""

from __future__ import annotations

import numpy as np


def _default_quantile_levels(n_quantiles: int) -> np.ndarray:
    """Generate default quantile levels when none are provided.

    The levels are evenly spaced in the open interval ``(0, 1)`` so that endpoints
    ``0`` and ``1`` are never used.

    Args:
        n_quantiles: Number of quantile levels required.

    Returns:
        One-dimensional array of length ``n_quantiles``. When ``n_quantiles <= 0``,
        returns an empty array.
    """
    if n_quantiles <= 0:
        return np.asarray([])
    return np.linspace(1.0 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)


def compute_ql(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    quantile_levels: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute per-horizon Quantile Loss.

    Formula (per horizon $t$):
        $\mathrm{QL}_t = \frac{1}{|Q|}\sum_{q \in Q}
        (q - \mathbb{1}\{y_t \le \hat{y}_t^{(q)}\})(y_t - \hat{y}_t^{(q)})$

    Args:
        predictions: Quantile prediction matrix with shape
            ``(prediction_length, n_quantiles)``.
        ground_truth: Ground-truth vector with shape ``(prediction_length,)``.
        quantile_levels: Quantile levels aligned with the second dimension of
            ``predictions``.
        mask: Optional boolean vector of length ``prediction_length``. Horizons where
            the mask is ``False`` are excluded from the loss and returned as ``NaN``.

    Returns:
        One-dimensional array of length ``prediction_length``. Each entry is the mean
        pinball loss across all quantile levels for that horizon.
    """
    y = ground_truth.reshape(-1, 1)
    q_hat = predictions
    q = quantile_levels.reshape(1, -1)
    indicator = (y <= q_hat).astype(float)
    loss = (q - indicator) * (y - q_hat)
    if mask is not None:
        valid = mask.reshape(-1, 1).astype(bool)
        loss = np.where(valid, loss, np.nan)

    valid_counts = np.sum(np.isfinite(loss), axis=1)
    loss_sum = np.nansum(loss, axis=1)
    result = np.full(loss.shape[0], np.nan, dtype=float)
    valid_rows = valid_counts > 0
    result[valid_rows] = loss_sum[valid_rows] / valid_counts[valid_rows]
    return result


def compute_sql(
    ql: np.ndarray,
    scales_by_hour: np.ndarray,
    season_length: int = 24,
    target_start_idx: int | None = None,
) -> np.ndarray:
    r"""Compute per-horizon Scaled Quantile Loss.

    Formula (per horizon $t+h$):
        $\mathrm{sQL}_{t+h} = \frac{\mathrm{QL}_{t+h}}{S^{(H_{t+h})}}$

    Args:
        ql: Per-horizon quantile loss vector with shape ``(prediction_length,)``.
        scales_by_hour: One-dimensional benchmark-global scale vector for the current
            feature. Its length must equal ``season_length``.
        season_length: Seasonal lag in hours. For the benchmark this is ``24``.
        target_start_idx: Absolute 0-based start index of the forecast window inside
            the full trajectory. Used to map each horizon to the proper hour-of-day
            bucket. When omitted, ``sQL`` is returned as all ``NaN``.

    Returns:
        One-dimensional array of length ``prediction_length``. Each entry is the
        corresponding ``QL`` value divided by the benchmark-global scale for that
        hour-of-day. Missing or zero scales yield ``NaN``.
    """
    if target_start_idx is None:
        return np.full_like(ql, np.nan, dtype=float)

    scales = np.asarray(scales_by_hour, dtype=float).reshape(-1)
    if scales.shape[0] != season_length:
        raise ValueError(f"scales_by_hour must have length {season_length}, got {scales.shape[0]}")

    target_indices = np.arange(target_start_idx, target_start_idx + ql.shape[0], dtype=int)
    scale_by_horizon = np.full(ql.shape, np.nan, dtype=float)
    valid_index_mask = target_indices >= 0
    scale_by_horizon[valid_index_mask] = scales[target_indices[valid_index_mask] % season_length]

    with np.errstate(divide="ignore", invalid="ignore"):
        sql = ql / scale_by_horizon
    return np.where(np.isfinite(scale_by_horizon) & (scale_by_horizon != 0.0), sql, np.nan)


def compute_quantiles_metrics(
    predictions: np.ndarray | None,
    ground_truth: np.ndarray,
    variable_names: list[str] | None = None,
    quantile_levels: np.ndarray | None = None,
    scales_by_feature: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    target_start_idx: int | None = None,
) -> dict[str, dict[str, list[float]] | None]:
    """Compute quantile-based forecasting metrics for all features in one sample.

    Args:
        predictions: Quantile prediction tensor with shape
            ``(n_features, prediction_length, n_quantiles)``. When ``None``, all
            metric groups are returned as ``None``.
        ground_truth: Ground-truth matrix with shape ``(n_features, prediction_length)``.
        variable_names: Optional feature names aligned with the first dimension. If
            omitted, default names ``var_0``, ``var_1``, ... are generated.
        quantile_levels: Quantile levels aligned with the last dimension of
            ``predictions``.
        scales_by_feature: Optional benchmark-global scale matrix with shape
            ``(n_features, 24)``. When provided, ``sQL`` is computed using the same
            benchmark-global scales as ``MASE``.
        mask: Optional observed-value mask with shape ``(n_features, prediction_length)``.
            Missing horizons are excluded from ``QL`` and ``sQL``.
        target_start_idx: Absolute 0-based forecast start index used to map horizons
            to hour-of-day buckets.

    Returns:
        A dictionary with the following keys:
        - ``ql``: maps each variable name to its per-horizon QL list.
        - ``sQL``: maps each variable name to its per-horizon scaled quantile loss.
        - ``ql_by_feature``: maps each variable name to the mean QL over its valid
          horizons.
        - ``ql_mean``: scalar mean of ``ql_by_feature`` across all features.
    """
    if predictions is None or quantile_levels is None:
        return {
            "ql": None,
            "sQL": None,
            "ql_by_feature": None,
            "ql_mean": None,
        }

    n_features = predictions.shape[0]
    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(n_features)]

    ql_per_feature: dict[str, list[float]] = {}
    sql_per_feature: dict[str, list[float]] = {}
    ql_by_feature: dict[str, float] = {}

    for i, var_name in enumerate(variable_names):
        feature_pred = predictions[i]
        feature_truth = ground_truth[i]
        feature_mask = mask[i] if mask is not None else None

        ql = compute_ql(feature_pred, feature_truth, quantile_levels, mask=feature_mask)
        if scales_by_feature is None:
            sql = np.full_like(ql, np.nan, dtype=float)
        else:
            sql = compute_sql(
                ql=ql,
                scales_by_hour=scales_by_feature[i],
                season_length=24,
                target_start_idx=target_start_idx,
            )

        ql_per_feature[var_name] = ql.tolist()
        sql_per_feature[var_name] = sql.tolist()
        ql_by_feature[var_name] = float(np.nanmean(ql))

    overall_ql = float(np.nanmean(np.asarray(list(ql_by_feature.values()), dtype=float)))

    return {
        "ql": ql_per_feature,
        "sQL": sql_per_feature,
        "ql_by_feature": ql_by_feature,
        "ql_mean": overall_ql,
    }
