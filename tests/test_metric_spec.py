"""Tests for the shared forecasting metric spec (single source of truth)."""

import math

import numpy as np
import pytest

from forecasting_evaluation.metrics import metric_spec as spec


def test_channel_groups():
    assert spec.CONTINUOUS_CHANNELS == tuple(range(0, 7))
    assert spec.SLEEP_CHANNELS == (7, 8)
    assert spec.WORKOUT_CHANNELS == tuple(range(9, 19))
    assert spec.BINARY_CHANNELS == tuple(range(7, 19))
    assert dict(spec.BINARY_GROUPS) == {"sleep": (7, 8), "workout": tuple(range(9, 19))}


def test_metric_direction():
    assert spec.metric_lower_is_better("mae") is True
    assert spec.metric_lower_is_better("auprc") is False
    with pytest.raises(ValueError):
        spec.metric_lower_is_better("nonsense")


def test_metric_to_error():
    # lower-is-better passes through; negative is invalid -> nan
    assert spec.metric_to_error("mae", 2.0) == 2.0
    assert math.isnan(spec.metric_to_error("mae", -1.0))
    # lower-is-better is NOT floored (perfect continuous error stays 0)
    assert spec.metric_to_error("mae", 0.0) == 0.0
    # higher-is-better flips to 1 - x; out-of-[0,1] -> nan
    assert spec.metric_to_error("auprc", 0.75) == pytest.approx(0.25)
    assert math.isnan(spec.metric_to_error("auprc", 1.5))
    with pytest.raises(ValueError):
        spec.metric_to_error("nonsense", 1.0)


def test_metric_to_error_binary_floor():
    # A perfect AUROC floors to ε instead of 0, so the paired skill ratio stays finite.
    assert spec.metric_to_error("auroc", 1.0) == pytest.approx(spec.BINARY_ERROR_FLOOR)
    assert spec.metric_to_error("auprc", 1.0) == pytest.approx(spec.BINARY_ERROR_FLOOR)
    # Just below perfect: 1 - 0.999 = 0.001 < ε -> floored to ε.
    assert spec.metric_to_error("auroc", 0.999) == pytest.approx(spec.BINARY_ERROR_FLOOR)
    # Comfortably above the floor: unchanged.
    assert spec.metric_to_error("auroc", 0.8) == pytest.approx(0.2)
    # NaN guard precedes the floor (single-class users stay NaN, get dropped).
    assert math.isnan(spec.metric_to_error("auroc", float("nan")))


def test_metric_channel_value():
    arr = np.array([[1.0, 3.0], [np.nan, np.nan]])
    assert spec.metric_channel_value(arr, 0) == 2.0  # mean over horizon
    assert math.isnan(spec.metric_channel_value(arr, 1))  # all-nan channel
    assert math.isnan(spec.metric_channel_value(arr, 5))  # out of range


def test_metric_channel_sum_count():
    # 2D (channel x horizon): sum + finite-cell count over the horizon
    arr = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan]])
    assert spec.metric_channel_sum_count(arr, 0) == (4.0, 2)  # 1+3 over 2 finite cells
    assert spec.metric_channel_sum_count(arr, 1) is None  # all-nan channel
    assert spec.metric_channel_sum_count(arr, 5) is None  # out of range
    # 1D (channel only): one cell per channel
    arr1d = np.array([2.0, np.nan])
    assert spec.metric_channel_sum_count(arr1d, 0) == (2.0, 1)
    assert spec.metric_channel_sum_count(arr1d, 1) is None  # nan
    # metric_channel_value delegates to sum/count
    assert spec.metric_channel_value(arr, 0) == pytest.approx(2.0)


def test_single_source_across_scripts():
    """skill + rank must share the spec's constants (no divergent definitions)."""
    from forecasting_evaluation.metrics import grouped_metric_rank_summary as rank
    from forecasting_evaluation.metrics import skill_score_summary as skill

    assert skill.LOWER_IS_BETTER_METRICS is spec.LOWER_IS_BETTER_METRICS
    assert skill.HIGHER_IS_BETTER_METRICS is spec.HIGHER_IS_BETTER_METRICS
    assert rank.DEFAULT_CONTINUOUS_CHANNELS is spec.CONTINUOUS_CHANNELS
    assert rank.DEFAULT_BINARY_GROUPS is spec.BINARY_GROUPS
    # helper aliases resolve to the spec implementations
    assert skill._metric_to_error is spec.metric_to_error
    assert rank._metric_lower_is_better is spec.metric_lower_is_better
