"""Gate A: the canonical per-user metrics substrate reconstructs every reducer's input.

Validates ``forecasting_evaluation.metrics.per_user_errors``:

  * ``write`` -> ``read`` round-trips the float64 ``metric_value`` + sidecar meta;
  * ``to_error_df`` reproduces the skill builder (``skill._build_error_table``,
    ``unit_id``) and the fairness builder (``fair._build_error_table``, ``user_id``);
  * ``to_rank_user_df`` reproduces ``rank._build_continuous_user_rows`` +
    ``rank._build_binary_user_rows`` (channel rows + sleep/activity group folds).

Uses 2D ``(channel x horizon)`` metric arrays with UNEVEN finite-cell counts per
window so the micro pooling that the substrate bakes is actually exercised; binary
channels include a pooled-1.0 channel so the ``BINARY_ERROR_FLOOR`` path is hit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting_evaluation.metrics import fairness_skill_score_summary as fair
from forecasting_evaluation.metrics import grouped_metric_rank_summary as rank
from forecasting_evaluation.metrics import skill_score_summary as skill
from forecasting_evaluation.metrics.per_user_errors import (
    build_per_user_metrics,
    read_per_user_metrics_parquet,
    to_error_df,
    to_rank_user_df,
    write_per_user_metrics_parquet,
)

N_CH = 9  # channels 0,1 (continuous) and 7,8 (binary) must be in range


def _win(channel_cells: dict[int, list[float]]) -> np.ndarray:
    """A ``(N_CH, horizon)`` window: listed channels filled (left), rest NaN."""
    horizon = max(len(cells) for cells in channel_cells.values())
    arr = np.full((N_CH, horizon), np.nan, dtype=float)
    for channel, cells in channel_cells.items():
        arr[channel, : len(cells)] = cells
    return arr


def _write_metric(root, metric, windows_by_user) -> None:
    """``{user: [ndarray(N_CH, horizon), ...]}`` -> ``<root>/<metric>/<user>.parquet`` tree."""
    mdir = root / metric
    mdir.mkdir(parents=True, exist_ok=True)
    for user_id, windows in windows_by_user.items():
        arrays = [[list(row) for row in w] for w in windows]
        tbl = pa.table(
            {
                "user_id": pa.array([user_id] * len(windows)),
                "history_length": pa.array([100 + i for i in range(len(windows))], pa.int64()),
                "forecasting_length": pa.array([24] * len(windows), pa.int64()),
                metric: pa.array(arrays, pa.list_(pa.list_(pa.float64()))),
            }
        )
        pq.write_table(tbl, mdir / f"{user_id}.parquet")


def _models(tmp_path, names):
    return {n: {"path": str(tmp_path / n), "display_name": n} for n in names}


# Uneven per-window cell counts so micro != macro; reused across metrics/models.
CONT = {
    "u0": [_win({0: [1, 1, 1, 1], 1: [2, 2]}), _win({0: [3], 1: [4, 4, 4]})],
    "u1": [_win({0: [2, 2], 1: [1]}), _win({0: [5, 5, 5, 5], 1: [3, 3]})],
}
BIN = {
    "u0": [_win({7: [0.8, 0.6], 8: [0.7]}), _win({7: [0.9], 8: [0.5, 0.5, 0.5]})],
    "u1": [_win({7: [0.55, 0.65], 8: [1.0, 1.0]}), _win({7: [0.95, 0.85], 8: [1.0]})],
}
CONT_METRICS = ["mae", "mse"]
BIN_METRICS = ["auroc", "auprc"]
CONT_CH = (0, 1)
BIN_CH = (7, 8)
CONT_GROUPS = [("act", (0, 1))]
BIN_GROUPS = [("sleep", (7, 8))]


def _setup(tmp_path):
    """Write metric trees for two models and build the substrate frame."""
    models = _models(tmp_path, ["m1", "m2"])
    for name in models:
        for metric in CONT_METRICS:
            _write_metric(tmp_path / name, metric, CONT)
        for metric in BIN_METRICS:
            _write_metric(tmp_path / name, metric, BIN)
    per_user = build_per_user_metrics(
        models=models,
        continuous_metrics=CONT_METRICS,
        binary_metrics=BIN_METRICS,
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
    )
    return models, per_user


def _assert_equiv(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    """Compare two frames ignoring row order / categorical-vs-str dtype, floats tight."""
    cols = list(expected.columns)
    a, e = actual[cols].copy(), expected[cols].copy()
    for df in (a, e):
        for col in cols:
            if str(df[col].dtype) in ("object", "category"):
                df[col] = df[col].astype(str)
    a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
    e = e.sort_values(keys, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, e, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_round_trip_preserves_values_and_meta(tmp_path):
    """Write -> read preserves the float64 metric_value and the sidecar meta."""
    models, per_user = _setup(tmp_path)
    out = tmp_path / "sub" / "per_user_errors.parquet"
    meta = {
        "within_user_aggregation": "micro",
        "aggregation_unit": "user",
        "models": list(models),
    }
    write_per_user_metrics_parquet(per_user, out, meta=meta)
    df2, meta2 = read_per_user_metrics_parquet(out)
    assert meta2 == meta
    assert str(df2["metric_value"].dtype) == "float64"
    _assert_equiv(df2, per_user, keys=["model", "group", "metric", "channel_idx", "user_id"])


def test_to_error_df_matches_skill_builder(tmp_path):
    """to_error_df(unit_id) reproduces skill._build_error_table row-for-row."""
    models, per_user = _setup(tmp_path)
    legacy = skill._build_error_table(
        models=models,
        metric_groups={
            "continuous": {"metrics": CONT_METRICS, "channel_indices": CONT_CH},
            "binary": {"metrics": BIN_METRICS, "channel_indices": BIN_CH},
        },
        aggregation_unit="user",
        within_user_aggregation="micro",
    )
    got = to_error_df(per_user, user_col="unit_id")
    _assert_equiv(got, legacy, keys=["model", "group", "metric", "channel_idx", "unit_id"])


def test_to_error_df_matches_fairness_builder(tmp_path):
    """to_error_df(user_id) reproduces fair._build_error_table row-for-row."""
    models, per_user = _setup(tmp_path)
    legacy = fair._build_error_table(
        models=models,
        continuous_metrics=CONT_METRICS,
        binary_metrics=BIN_METRICS,
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        within_user_aggregation="micro",
    )
    got = to_error_df(per_user, user_col="user_id")
    _assert_equiv(got, legacy, keys=["model", "group", "metric", "channel_idx", "user_id"])


def test_to_rank_user_df_matches_legacy_builders(tmp_path):
    """to_rank_user_df reproduces the legacy continuous+binary rank builders."""
    models, per_user = _setup(tmp_path)
    continuous_user = rank._build_continuous_user_rows(
        models=models,
        metrics=CONT_METRICS,
        channel_indices=CONT_CH,
        within_user_aggregation="micro",
        groups=CONT_GROUPS,
    )
    binary_user = rank._build_binary_user_rows(
        models=models,
        metrics=BIN_METRICS,
        groups=BIN_GROUPS,
        within_user_aggregation="micro",
    )
    legacy = pd.concat([continuous_user, binary_user], ignore_index=True)
    got = to_rank_user_df(per_user, binary_groups=BIN_GROUPS, continuous_groups=CONT_GROUPS)
    _assert_equiv(
        got,
        legacy,
        keys=["model", "scope_type", "scope", "metric", "channel_idx", "user_id"],
    )


def test_build_per_user_metrics_rejects_macro(tmp_path):
    """The substrate producer refuses macro (not reconstructable for binary)."""
    models, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="micro"):
        build_per_user_metrics(
            models=models,
            continuous_metrics=CONT_METRICS,
            binary_metrics=BIN_METRICS,
            continuous_channel_indices=CONT_CH,
            binary_channel_indices=BIN_CH,
            within_user_aggregation="macro",
        )


def test_public_api_skill_path_matches_trees(tmp_path):
    """Public-API skill path (per-method substrates -> concat -> reducer) == from-trees skill.

    Mirrors what ``evaluate_forecasting`` does: build the method's substrate and the
    baseline's substrate independently (paper set: mae + auroc), concat them, and run
    ``compute_skill_score_tables`` over the combined substrate. The model-summary must
    equal computing skill straight from the metric trees for the same two models.
    """
    for name in ("m", "seasonal_naive"):
        _write_metric(tmp_path / name, "mae", CONT)
        _write_metric(tmp_path / name, "auroc", BIN)
    models = _models(tmp_path, ["m", "seasonal_naive"])
    paper = dict(continuous_metrics=["mae"], binary_metrics=["auroc"])
    sub_m = build_per_user_metrics(
        models={"m": models["m"]},
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        **paper,
    )
    sub_b = build_per_user_metrics(
        models={"seasonal_naive": models["seasonal_naive"]},
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        **paper,
    )
    combined = pd.concat([sub_m, sub_b], ignore_index=True)
    common = dict(
        models=models,
        baseline_model="seasonal_naive",
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        clip_lower=0.01,
        clip_upper=100.0,
        min_pairs=1,
        aggregation_unit="user",
        within_user_aggregation="micro",
        **paper,
    )
    _, summ_sub, _ = skill.compute_skill_score_tables(**common, per_user_metrics=combined)
    _, summ_trees, _ = skill.compute_skill_score_tables(**common)
    _assert_equiv(summ_sub, summ_trees, keys=["model"])
