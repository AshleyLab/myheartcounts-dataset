"""Helpers for merging selected forecasting channels before offline metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_CHANNEL_MERGE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 4),
)

_METRIC_KEYS = ("mae", "mse", "mase", "mase_all", "ql", "sql")


@dataclass(frozen=True)
class ResolvedChannelMerges:
    """Resolved channel merge plan for one feature layout."""

    merge_pairs: tuple[tuple[int, int], ...]
    zero_feature_indices: tuple[int, ...]


def resolve_channel_merges(
    variable_names: list[str],
    merge_pairs: tuple[tuple[int, int], ...] = DEFAULT_CHANNEL_MERGE_PAIRS,
    *,
    combine_channels: bool = True,
) -> ResolvedChannelMerges:
    """Resolve a fixed merge plan against the current feature layout length."""
    if not combine_channels:
        return ResolvedChannelMerges(merge_pairs=(), zero_feature_indices=())

    resolved_pairs: list[tuple[int, int]] = []
    zero_feature_indices: list[int] = []
    feature_count = len(variable_names)

    for primary_idx, secondary_idx in merge_pairs:
        if primary_idx < 0 or secondary_idx < 0:
            continue
        if primary_idx >= feature_count or secondary_idx >= feature_count:
            continue
        if primary_idx == secondary_idx:
            continue
        resolved_pairs.append((int(primary_idx), int(secondary_idx)))
        zero_feature_indices.append(int(secondary_idx))

    return ResolvedChannelMerges(
        merge_pairs=tuple(resolved_pairs),
        zero_feature_indices=tuple(sorted(set(zero_feature_indices))),
    )


def merge_channel_first_array(
    values: np.ndarray | None,
    merge_plan: ResolvedChannelMerges,
    *,
    zero_fill_value: float = 0.0,
) -> np.ndarray | None:
    """Merge a channel-first array using pointwise nan-aware averaging."""
    if values is None:
        return None

    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim < 2:
        return arr

    for primary_idx, secondary_idx in merge_plan.merge_pairs:
        primary_values = arr[primary_idx]
        secondary_values = arr[secondary_idx]
        arr[primary_idx] = _pointwise_nanmean(primary_values, secondary_values)
        arr[secondary_idx] = np.full(arr[secondary_idx].shape, float(zero_fill_value), dtype=float)

    return arr


def zero_out_metrics_output(
    metrics_output: dict[str, Any],
    zero_feature_indices: tuple[int, ...],
    metric_keys: tuple[str, ...] = _METRIC_KEYS,
) -> dict[str, Any]:
    """Force merged-away feature rows to all zeros in saved metric outputs."""
    if not zero_feature_indices:
        return metrics_output

    updated = dict(metrics_output)
    for metric_key in metric_keys:
        metric_value = updated.get(metric_key)
        updated[metric_key] = zero_out_channel_rows(metric_value, zero_feature_indices)
    return updated


def zero_out_channel_rows(
    metric_value: Any,
    zero_feature_indices: tuple[int, ...],
) -> Any:
    """Set selected feature rows to zero while preserving the original nested shape."""
    if metric_value is None:
        return None

    arr = np.asarray(metric_value, dtype=float)
    if arr.ndim != 2:
        return metric_value

    arr = arr.copy()
    for feature_idx in zero_feature_indices:
        if 0 <= feature_idx < arr.shape[0]:
            arr[feature_idx] = 0.0

    return arr.tolist()


def _pointwise_nanmean(primary_values: np.ndarray, secondary_values: np.ndarray) -> np.ndarray:
    """Average two aligned arrays pointwise while ignoring missing values."""
    primary_arr = np.asarray(primary_values, dtype=float)
    secondary_arr = np.asarray(secondary_values, dtype=float)
    if primary_arr.shape != secondary_arr.shape:
        raise ValueError(
            "Merged channels must share the same shape, "
            f"got {primary_arr.shape} and {secondary_arr.shape}"
        )

    stacked = np.stack([primary_arr, secondary_arr], axis=0)
    valid_mask = np.isfinite(stacked)
    valid_count = valid_mask.sum(axis=0)
    summed = np.where(valid_mask, stacked, 0.0).sum(axis=0)

    merged = np.full(primary_arr.shape, np.nan, dtype=float)
    non_empty_mask = valid_count > 0
    merged[non_empty_mask] = summed[non_empty_mask] / valid_count[non_empty_mask]
    return merged
