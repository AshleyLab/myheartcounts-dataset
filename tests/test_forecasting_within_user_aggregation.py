"""Within-user micro vs macro aggregation across the forecasting readers.

Uses 2D ``(channel x horizon)`` metric arrays with UNEVEN finite-cell counts per
window so macro (mean of per-window means) and micro (sum/count over all finite
horizon cells across a user's windows) differ. The equal-count fixtures in the
other suites cannot tell the two apart.

Worked example used throughout — one user, channel 0, two windows:
    window 1: [1, 1, 1, 1]  -> sum 4, count 4, mean 1.0
    window 2: [3]           -> sum 3, count 1, mean 3.0
    macro E = mean(1.0, 3.0)      = 2.0   (n_values = 2 windows)
    micro E = (4 + 3) / (4 + 1)   = 1.4   (n_values = 5 finite cells)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting_evaluation.metrics import fair_skill_score
from forecasting_evaluation.metrics import fairness_skill_score_summary as fair
from forecasting_evaluation.metrics import grouped_metric_rank_summary as rank
from forecasting_evaluation.metrics import skill_score_summary as skill
from forecasting_evaluation.metrics.bootstrap_skill_rank import _draw_replica_frame, _resample

N_CH = 2


def _ch0(*cells: float) -> np.ndarray:
    """A ``(N_CH, len(cells))`` window with channel 0 = ``cells`` and others NaN."""
    arr = np.full((N_CH, len(cells)), np.nan, dtype=float)
    arr[0, :] = cells
    return arr


def _write_metric(root, metric, windows_by_user) -> None:
    """``windows_by_user``: ``{user_id: [ndarray(N_CH, horizon), ...]}`` -> parquet tree."""
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


def test_skill_reader_macro_vs_micro(tmp_path):
    """The skill reader returns macro (window-mean) vs micro (cell-pooled) errors and counts."""
    _write_metric(tmp_path / "m", "mae", {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)]})
    common = dict(
        model_name="m",
        model_root=str(tmp_path / "m"),
        metric_name="mae",
        channel_indices=(0,),
        group_name="continuous",
        aggregation_unit="user",
    )
    macro = skill._load_metric_values(**common, within_user_aggregation="macro")
    micro = skill._load_metric_values(**common, within_user_aggregation="micro")
    mac = macro.set_index("channel_idx").loc[0]
    mic = micro.set_index("channel_idx").loc[0]
    assert mac["error"] == pytest.approx(2.0)
    assert int(mac["n_values"]) == 2  # windows
    assert mic["error"] == pytest.approx(1.4)
    assert int(mic["n_values"]) == 5  # finite cells


def test_rank_reader_macro_vs_micro(tmp_path):
    """The rank reader's metric_value differs by mode while n_values stays the cell count."""
    _write_metric(tmp_path / "m", "mae", {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)]})
    common = dict(
        model_name="m",
        model_root=str(tmp_path / "m"),
        metric_name="mae",
        channel_indices=(0,),
        scope_type="continuous_channel",
    )
    macro = rank._load_channel_user_metrics(**common, within_user_aggregation="macro")
    micro = rank._load_channel_user_metrics(**common, within_user_aggregation="micro")
    assert macro.set_index("channel_idx").loc[0, "metric_value"] == pytest.approx(2.0)
    assert micro.set_index("channel_idx").loc[0, "metric_value"] == pytest.approx(1.4)
    # n_values reports the total finite-cell count regardless of mode.
    assert int(macro.set_index("channel_idx").loc[0, "n_values"]) == 5
    assert int(micro.set_index("channel_idx").loc[0, "n_values"]) == 5


def test_fairness_reader_macro_vs_micro(tmp_path):
    """The fairness error table yields macro vs micro per-user errors from uneven windows."""
    _write_metric(tmp_path / "m", "mae", {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)]})
    common = dict(
        models=_models(tmp_path, ["m"]),
        continuous_metrics=["mae"],
        binary_metrics=[],
        continuous_channel_indices=(0,),
        binary_channel_indices=(),
    )
    macro = fair._build_error_table(**common, within_user_aggregation="macro")
    micro = fair._build_error_table(**common, within_user_aggregation="micro")
    assert macro.set_index("channel_idx").loc[0, "error"] == pytest.approx(2.0)
    assert micro.set_index("channel_idx").loc[0, "error"] == pytest.approx(1.4)


def _skill_summary(models, within):
    _, summary, _ = skill.compute_skill_score_tables(
        models=models,
        baseline_model="baseline",
        continuous_metrics=["mae"],
        binary_metrics=[],
        continuous_channel_indices=(0,),
        binary_channel_indices=(),
        clip_lower=0.01,
        clip_upper=100.0,
        min_pairs=1,
        aggregation_unit="user",
        within_user_aggregation=within,
    )
    return summary.set_index("model")


def test_skill_tables_macro_vs_micro(tmp_path):
    """End-to-end skill score differs by mode (0.0 macro vs 0.3 micro) for uneven counts."""
    # baseline is uniform (E_b = 2.0 under both modes); only the model has uneven counts.
    _write_metric(tmp_path / "baseline", "mae", {"u0": [_ch0(2.0, 2.0), _ch0(2.0, 2.0)]})
    _write_metric(tmp_path / "good", "mae", {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)]})
    models = _models(tmp_path, ["baseline", "good"])
    # macro: E_m = 2.0 -> ratio 1.0 -> skill 0.0; micro: E_m = 1.4 -> ratio 0.7 -> skill 0.3.
    assert _skill_summary(models, "macro").loc["good", "channel_0_score"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert _skill_summary(models, "micro").loc["good", "channel_0_score"] == pytest.approx(
        0.3, abs=1e-9
    )


def test_skill_default_is_micro(tmp_path):
    """Omitting the toggle uses micro (the new default)."""
    _write_metric(tmp_path / "baseline", "mae", {"u0": [_ch0(2.0, 2.0), _ch0(2.0, 2.0)]})
    _write_metric(tmp_path / "good", "mae", {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)]})
    models = _models(tmp_path, ["baseline", "good"])
    _, summary, _ = skill.compute_skill_score_tables(
        models=models,
        baseline_model="baseline",
        continuous_metrics=["mae"],
        binary_metrics=[],
        continuous_channel_indices=(0,),
        binary_channel_indices=(),
        clip_lower=0.01,
        clip_upper=100.0,
        min_pairs=1,
        aggregation_unit="user",
    )
    assert summary.set_index("model").loc["good", "channel_0_score"] == pytest.approx(0.3, abs=1e-9)


def test_bootstrap_identity_matches_point_both_modes(tmp_path):
    """Identity draw reproduces the point estimate under BOTH modes; macro != micro."""
    _write_metric(
        tmp_path / "baseline",
        "mae",
        {"u0": [_ch0(2.0, 2.0), _ch0(2.0, 2.0)], "u1": [_ch0(2.0, 2.0, 2.0)]},
    )
    _write_metric(
        tmp_path / "good",
        "mae",
        {
            "u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)],
            "u1": [_ch0(1.0, 1.0), _ch0(5.0, 5.0, 5.0, 5.0)],
        },
    )
    models = _models(tmp_path, ["baseline", "good"])
    metric_groups = {
        "continuous": {"metrics": ["mae"], "channel_indices": (0,)},
        "binary": {"metrics": [], "channel_indices": ()},
    }

    point_scores = {}
    for within in ("macro", "micro"):
        error_df = skill._build_error_table(
            models=models,
            metric_groups=metric_groups,
            aggregation_unit="user",
            within_user_aggregation=within,
        )
        users = sorted(set(error_df["unit_id"].astype(str)))
        replicas = _draw_replica_frame(users, np.arange(len(users)))  # identity draw
        err_b = _resample(error_df, replicas, "unit_id")
        long_b = skill._compute_long_skill_scores(
            error_df=err_b,
            models=models,
            baseline_model="baseline",
            clip_lower=0.01,
            clip_upper=100.0,
            min_pairs=1,
        )
        summ_b = skill._build_model_summary(
            long_df=long_b, models=models, baseline_model="baseline"
        )
        point = _skill_summary(models, within)
        draw_score = summ_b.set_index("model").loc["good", "channel_0_score"]
        point_score = point.loc["good", "channel_0_score"]
        assert draw_score == pytest.approx(point_score, rel=1e-9, abs=1e-12)
        point_scores[within] = point_score

    assert point_scores["macro"] != pytest.approx(point_scores["micro"], abs=1e-6)


def _fair_overall(error_df, demographics, model):
    """Disparity-ratio fair skill score (``overall`` scope) for one model."""
    out = fair_skill_score.compute_fair_skill_scores_from_errors(
        error_df,
        demographics,
        attrs=("age_group",),
        baseline_method="baseline",
    )
    row = out[(out["model"] == model) & (out["scope"] == "overall")]
    return float(row["fair_skill_score"].iloc[0])


def test_fair_skill_score_macro_vs_micro(tmp_path):
    """End-to-end disparity-ratio fair skill score moves between micro and macro.

    Counterpart to ``test_skill_tables_macro_vs_micro`` for the fairness metric:
    the uneven-count fixtures only differed at ``_build_error_table`` before; this
    drives the full disparity-ratio score. ``age_group`` has two subgroups, and
    only the model's *young* user has uneven per-window cell counts, so its
    subgroup error — hence the model gap ``D_m``, hence the ratio against the
    baseline gap ``D_b`` — depends on the mode:

        young model error: macro 2.0, micro 1.4   (windows [1,1,1,1] and [3])
        old   model error: 4.0 (uniform, both modes)
        baseline gap D_b = |4.0 - 2.0| = 2.0 (uniform, both modes)
        macro: D_m = 2.0 -> ratio 1.0 -> S = 0.0
        micro: D_m = 2.6 -> ratio 1.3 -> S = -0.3
    """
    # u0 -> young, u1 -> old; both models score both users (paired cohort).
    _write_metric(tmp_path / "baseline", "mae", {"u0": [_ch0(2.0, 2.0)], "u1": [_ch0(4.0, 4.0)]})
    _write_metric(
        tmp_path / "good",
        "mae",
        {"u0": [_ch0(1.0, 1.0, 1.0, 1.0), _ch0(3.0)], "u1": [_ch0(4.0, 4.0)]},
    )
    models = _models(tmp_path, ["baseline", "good"])
    demographics = {"u0": {"age_group": "young"}, "u1": {"age_group": "old"}}
    common = dict(
        models=models,
        continuous_metrics=["mae"],
        binary_metrics=[],
        continuous_channel_indices=(0,),
        binary_channel_indices=(),
    )
    macro_err = fair._build_error_table(**common, within_user_aggregation="macro")
    micro_err = fair._build_error_table(**common, within_user_aggregation="micro")

    assert _fair_overall(macro_err, demographics, "good") == pytest.approx(0.0, abs=1e-9)
    assert _fair_overall(micro_err, demographics, "good") == pytest.approx(-0.3, abs=1e-9)
    # The baseline's self-ratio is 1, so its fair skill score is 0 under both modes.
    assert _fair_overall(macro_err, demographics, "baseline") == pytest.approx(0.0, abs=1e-9)
    assert _fair_overall(micro_err, demographics, "baseline") == pytest.approx(0.0, abs=1e-9)
