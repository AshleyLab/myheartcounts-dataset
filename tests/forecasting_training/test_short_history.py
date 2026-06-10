"""Short-history windows: included + NaN-left-padded by default, dropped when off.

Reproduces the train/eval distribution-equivalence claim at the dataset level:
with ``include_short_history=True`` the dataset keeps windows whose history is
shorter than ``n_steps`` and left-pads them to the fixed window (leading
positions missing), exactly as ``BasePyPOTSForecastingModel.predict`` does at
eval time.
"""

from __future__ import annotations

import torch

from forecasting_evaluation.data.online_dataset import (
    ForecastingRowGroup,
    ForecastingWindowDescriptor,
    ModelConfig,
    PyPOTSForecastingDataset,
)

N_FEATURES = 2
N_STEPS = 168
N_PRED = 24
TRAJ_LEN = 300
SHORT_DAY = 1  # history_end_hour = 24  -> shorter than n_steps (168)
FULL_DAY = 8  # history_end_hour = 192 -> full window


def _make_dataset(include_short_history: bool) -> PyPOTSForecastingDataset:
    history_cf = torch.arange(N_FEATURES * TRAJ_LEN, dtype=torch.float32).reshape(
        N_FEATURES, TRAJ_LEN
    )
    windows = (
        ForecastingWindowDescriptor(current_day=SHORT_DAY, history_end_hour=24, pred_end_hour=48),
        ForecastingWindowDescriptor(current_day=FULL_DAY, history_end_hour=192, pred_end_hour=216),
    )
    row_groups = [ForecastingRowGroup(dataset_row_idx=0, user_id="u", windows=windows)]
    model_config = ModelConfig(n_steps=N_STEPS, n_pred_steps=N_PRED, n_features=N_FEATURES)
    return PyPOTSForecastingDataset(
        history_cf_source=[history_cf],
        row_groups=row_groups,
        model_config=model_config,
        daily_start_hour_offset=0,
        include_short_history=include_short_history,
    )


def test_short_history_included_by_default() -> None:
    ds = _make_dataset(include_short_history=True)
    assert len(ds) == 2  # short + full


def test_short_history_dropped_when_disabled() -> None:
    ds = _make_dataset(include_short_history=False)
    assert len(ds) == 1  # only the full (day-8) window survives


def test_short_window_is_nan_left_padded() -> None:
    ds = _make_dataset(include_short_history=True)
    # Sample 0 = first window of the first row group = the short (day-1) window.
    _idx, x, missing_mask, x_pred, _x_pred_mask = ds._fetch_data_from_manifest(0)

    assert tuple(x.shape) == (N_STEPS, N_FEATURES)
    assert tuple(x_pred.shape) == (N_PRED, N_FEATURES)

    pad = N_STEPS - 24  # 144 leading positions are missing (older context)
    assert torch.all(missing_mask[:pad] == 0)
    assert torch.all(missing_mask[pad:] == 1)
    # fill_and_get_mask_torch fills NaN -> 0, so the padded region is exactly 0.
    assert torch.all(x[:pad] == 0)
    # Observed tail carries real (finite, non-zero) history.
    assert torch.all(missing_mask[pad:] == 1)


def test_full_window_has_no_padding() -> None:
    ds = _make_dataset(include_short_history=True)
    # Sample 1 = the full (day-8) window: no missing rows from padding.
    _idx, x, missing_mask, _x_pred, _m = ds._fetch_data_from_manifest(1)
    assert tuple(x.shape) == (N_STEPS, N_FEATURES)
    assert torch.all(missing_mask == 1)
