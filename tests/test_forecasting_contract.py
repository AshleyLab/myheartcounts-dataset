"""Tests for the unified forecasting-model contract (Track 3).

Covers:
- the single duck-typed harness call path (``_invoke_forecaster`` +
  signature-based kwarg forwarding + return normalization),
- the model-agnostic eval window set (manifest no longer drops short-history
  windows for fixed-context models),
- the Seasonal-Naive fallback substitution for NaN predictions,
- (data-gated) end-to-end fallback visibility through ``evaluate_forecasting``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from forecasting_evaluation.evaluation.evaluator import (
    ForecastingEvaluator,
    _forward_kwargs,
    _invoke_forecaster,
    _normalize_forecast_output,
)
from forecasting_evaluation.forecasting_training.online_dataset import (
    ForecastingSampleIndexBuilder,
)
from forecasting_evaluation.forecasting_training.standard_scaler import (
    ChannelStandardScalerStats,
)
from forecasting_evaluation.models.deep_learning_model.pypots_forecasting_base import (
    BasePyPOTSForecastingModel,
)
from forecasting_evaluation.models.naive.seasonal_naive import SeasonalNaiveModel


# --------------------------------------------------------------------------- #
# Unified harness call path
# --------------------------------------------------------------------------- #
class _MinimalForecaster:
    """Declares only (history, horizon); returns a bare point array."""

    def predict(self, history, horizon):
        return np.zeros((history.shape[0], horizon), dtype=np.float32)


class _RichForecaster:
    """Declares optional metadata kwargs and returns a (point, quantiles) tuple."""

    def __init__(self):
        self.seen_kwargs: dict = {}

    def predict(self, history, horizon, *, variable_names=None, index_days=None):
        self.seen_kwargs = {"variable_names": variable_names, "index_days": index_days}
        point = np.zeros((history.shape[0], horizon), dtype=np.float32)
        quantiles = np.zeros((history.shape[0], horizon, 3), dtype=np.float32)
        return point, quantiles


class _KwargsForecaster:
    """Declares **kwargs; should receive every candidate metadata kwarg."""

    def __init__(self):
        self.seen_kwargs: dict = {}

    def predict(self, history, horizon, **kwargs):
        self.seen_kwargs = dict(kwargs)
        return np.zeros((history.shape[0], horizon), dtype=np.float32)


def _meta():
    return {
        "variable_names": ["a", "b"],
        "past_covariates": None,
        "future_covariates": None,
        "index_days": 3,
    }


def test_forward_kwargs_only_declared():
    assert _forward_kwargs(_MinimalForecaster(), _meta()) == {}
    assert _forward_kwargs(_RichForecaster(), _meta()) == {
        "variable_names": ["a", "b"],
        "index_days": 3,
    }
    assert _forward_kwargs(_KwargsForecaster(), _meta()) == _meta()


def test_normalize_forecast_output_variants():
    arr = np.zeros((2, 4), dtype=np.float32)
    point, quant = _normalize_forecast_output(arr)
    assert quant is None and point.shape == (2, 4)

    point, quant = _normalize_forecast_output((arr, None))
    assert quant is None and point.shape == (2, 4)

    q = np.zeros((2, 4, 3), dtype=np.float32)
    point, quant = _normalize_forecast_output((arr, q))
    assert quant is not None and quant.shape == (2, 4, 3)


def test_invoke_forecaster_minimal():
    history = np.ones((2, 50), dtype=np.float32)
    point, quant, perf = _invoke_forecaster(_MinimalForecaster(), history, 24, _meta())
    assert point.shape == (2, 24)
    assert quant is None
    assert "prediction_time_seconds" in perf and "memory_usage_mb" in perf


def test_invoke_forecaster_forwards_declared_kwargs():
    model = _RichForecaster()
    history = np.ones((2, 50), dtype=np.float32)
    point, quant, _perf = _invoke_forecaster(model, history, 24, _meta())
    assert point.shape == (2, 24)
    assert quant.shape == (2, 24, 3)
    assert model.seen_kwargs == {"variable_names": ["a", "b"], "index_days": 3}


# --------------------------------------------------------------------------- #
# Seasonal-Naive fallback substitution
# --------------------------------------------------------------------------- #
def test_seasonal_naive_fallback_fills_nan_only():
    rng = np.random.default_rng(0)
    history = rng.normal(size=(3, 72)).astype(np.float32)
    horizon = 24
    fallback = SeasonalNaiveModel(seed=42, seasonal=24)

    point = rng.normal(size=(3, horizon)).astype(np.float32)
    point[1, :] = np.nan  # channel 1 cannot be predicted
    point[2, 5] = np.nan  # one cell on channel 2

    repaired, mask = ForecastingEvaluator._apply_seasonal_naive_fallback(
        point_result=point.copy(),
        history_window=history,
        horizon=horizon,
        n_channels=3,
        fallback_model=fallback,
    )
    # Mask marks exactly the NaN positions.
    assert mask[1, :].all()
    assert mask[2, 5]
    assert mask.sum() == horizon + 1
    # Repaired output is finite where the seasonal-naive baseline is finite, and
    # untouched (equal) where the model already produced a value.
    finite_model = np.isfinite(point)
    assert np.array_equal(repaired[finite_model], point[finite_model])


def test_seasonal_naive_fallback_all_none_prediction():
    history = np.ones((4, 48), dtype=np.float32)
    fallback = SeasonalNaiveModel(seed=42, seasonal=24)
    repaired, mask = ForecastingEvaluator._apply_seasonal_naive_fallback(
        point_result=None,
        history_window=history,
        horizon=12,
        n_channels=4,
        fallback_model=fallback,
    )
    assert repaired.shape == (4, 12)
    assert mask.shape == (4, 12)
    assert mask.all()  # everything was a gap


# --------------------------------------------------------------------------- #
# Model-agnostic eval window set
# --------------------------------------------------------------------------- #
def _write_synthetic_split_and_index(tmp_path):
    import datasets as hf_ds

    # u1: 200 hours (days 1..7 forecastable for horizon 24); u2: 150 hours.
    ds = hf_ds.Dataset.from_dict(
        {
            "user_id": ["u1", "u2"],
            "values": [[0.0] * 200, [0.0] * 150],
        }
    )
    sample_index = {"u1": [1, 2, 3, 4, 5, 6, 7], "u2": [1, 2, 3, 4]}
    index_path = tmp_path / "sample_index.json"
    index_path.write_text(json.dumps(sample_index))
    return ds, index_path


def _row_group_signature(row_groups):
    return [
        (
            rg.user_id,
            tuple((w.current_day, w.history_end_hour, w.pred_end_hour) for w in rg.windows),
        )
        for rg in row_groups
    ]


def test_manifest_window_set_is_model_agnostic(tmp_path):
    """The window set must not depend on a model's fixed context length."""
    ds, index_path = _write_synthetic_split_and_index(tmp_path)

    short_ctx = ForecastingSampleIndexBuilder(
        split_ds=ds, sample_index_file=index_path, n_steps=1, n_pred_steps=24
    ).build_row_groups()
    long_ctx = ForecastingSampleIndexBuilder(
        split_ds=ds, sample_index_file=index_path, n_steps=168, n_pred_steps=24
    ).build_row_groups()

    assert _row_group_signature(short_ctx) == _row_group_signature(long_ctx)
    # Sanity: short-history windows (e.g. day 1, history_end=24 < 168) are present
    # for the long-context config — they are no longer dropped.
    sig = dict(_row_group_signature(long_ctx))
    assert (1, 24, 48) in sig["u1"]


# --------------------------------------------------------------------------- #
# PyPOTS internal standardization (raw history in -> raw forecast out)
# --------------------------------------------------------------------------- #
class _EchoModel:
    """Fake PyPOTS backbone that echoes the last ``n_pred`` context steps.

    This *fake* backbone intentionally **propagates** NaN (it echoes the padded
    input verbatim) to exercise the adapter's padding-shape logic. Real PyPOTS
    backbones fill NaN internally (``fill_and_get_mask_torch``) and emit finite
    forecasts — they do not return NaN for short history.
    """

    def __init__(self, n_pred: int):
        self.n_pred = n_pred

    def predict(self, batch: dict) -> dict:
        x = batch["X"]  # (1, n_steps, n_features), standardized
        return {"forecasting": x[:, -self.n_pred :, :]}


class _FakePyPOTS(BasePyPOTSForecastingModel):
    """PyPOTS adapter wired to a fake backbone, bypassing checkpoint loading."""

    def __init__(self, scaler_stats, n_steps: int, n_pred: int):
        self._scaler_stats = scaler_stats
        self._ns = n_steps
        self._np = n_pred
        self._model = _EchoModel(n_pred)

    @property
    def n_steps(self) -> int:
        return self._ns

    @property
    def n_pred_steps(self) -> int:
        return self._np

    def build_model(self):  # pragma: no cover - not used in this test
        raise NotImplementedError


def test_pypots_predict_standardizes_raw_history_and_inverts():
    """Harness feeds raw history; model standardizes in and inverts out.

    With a backbone that echoes the trailing context, the round-trip
    (standardize -> slice -> echo -> inverse) must return the *raw* values at
    the corresponding timesteps, proving internal scaling is wired correctly.
    """
    n_features, n_steps, n_pred = 3, 48, 24
    means = np.array([10.0, -5.0, 100.0])
    stds = np.array([2.0, 4.0, 50.0])
    stats = ChannelStandardScalerStats(
        means=means, stds=stds, valid_counts=np.full(n_features, 1000)
    )
    model = _FakePyPOTS(stats, n_steps=n_steps, n_pred=n_pred)

    rng = np.random.default_rng(0)
    raw_history = rng.normal(size=(n_features, 200)).astype(np.float32) * 30 + 50

    point, quant = model.predict(raw_history, n_pred)
    assert quant is None
    assert point.shape == (n_features, n_pred)
    # Echo of trailing context, round-tripped through scaling, recovers raw values.
    np.testing.assert_allclose(point, raw_history[:, -n_pred:], rtol=1e-4, atol=1e-3)


def test_pypots_predict_left_pads_short_history():
    """A history shorter than n_steps is left-padded (NaN) and still predicts."""
    n_features, n_steps, n_pred = 2, 48, 12
    stats = ChannelStandardScalerStats(
        means=np.zeros(n_features), stds=np.ones(n_features), valid_counts=np.full(n_features, 10)
    )
    model = _FakePyPOTS(stats, n_steps=n_steps, n_pred=n_pred)

    # Only 6 real timesteps (< n_pred): the trailing echo window overlaps the
    # NaN left-pad, so some forecast cells are NaN (model "cannot predict" them).
    raw_history = np.ones((n_features, 6), dtype=np.float32)
    point, _quant = model.predict(raw_history, n_pred)
    assert point.shape == (n_features, n_pred)
    # The most recent cells (backed by real history) are finite.
    assert np.isfinite(point[:, -1]).all()


# --------------------------------------------------------------------------- #
# Data-gated end-to-end fallback visibility
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("MHC_DATA_DIR"),
    reason="requires MHC_DATA_DIR with the full dataset",
)
def test_fallback_visible_end_to_end():
    import openmhc

    class _GappyForecaster:
        model_name = "gappy_test"

        def predict(self, history, horizon):
            filled = np.where(np.isfinite(history), history, 0.0).astype(np.float32)
            last = (
                filled[:, -1:]
                if filled.shape[1]
                else np.zeros((filled.shape[0], 1), np.float32)
            )
            out = np.tile(last, (1, horizon)).astype(np.float32)
            out[:3, :] = np.nan  # declare channels 0,1,2 unpredictable
            return out

    res = openmhc.evaluate_forecasting(
        _GappyForecaster(), version="full", forecasting_length=24, max_samples=5, seed=42
    )
    assert 0.0 < res.overall_fallback_rate < 1.0
    for ch in ("ch_0", "ch_1", "ch_2"):
        assert res.fallback_rate.get(ch) == pytest.approx(1.0)
        # Channels the model could not predict are still scored (filled), not dropped.
        assert ch in res.per_channel
        assert np.isfinite(res.per_channel[ch]["mae"])
