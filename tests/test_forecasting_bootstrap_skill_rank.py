"""Tests for the forecasting paired user-bootstrap of skill score + mean rank.

A tiny on-disk metrics fixture is written in the real layout
``<model>/<metric>/<user>.parquet`` (per-channel metric arrays), then the
bootstrap is exercised against the existing point-flow aggregators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting_evaluation.metrics.bootstrap_skill_rank import (
    _build_error_table,
    _build_model_summary,
    _compute_long_skill_scores,
    _draw_replica_frame,
    _resample,
    bootstrap_skill_rank,
)
from forecasting_evaluation.metrics.grouped_metric_rank_summary import (
    _compute_overall_category_balanced_ranks,
)
from forecasting_evaluation.metrics.skill_score_summary import (
    _aggregate_overall_category_balanced_score,
    compute_skill_score_tables,
)

# Per-channel base values per model (channels 0-1 continuous via mae,
# channels 7-8 binary via auprc). "good" beats "baseline" beats "bad".
_MAE = {"baseline": 1.0, "good": 0.5, "bad": 2.0}
_AUPRC = {"baseline": 0.6, "good": 0.8, "bad": 0.4}

CONT_CH = (0, 1)
BIN_CH = (7, 8)
N_CH = 19
N_USERS = 12
WINDOWS_PER_USER = 3


def _write_metric_file(path, *, metric, model, n_users, seed):
    rng = np.random.default_rng(seed)
    uids, hls, fls, arrays = [], [], [], []
    for u in range(n_users):
        for w in range(WINDOWS_PER_USER):
            arr = np.full(N_CH, np.nan, dtype=float)
            if metric == "mae":
                base = _MAE[model] * (1.0 + 0.4 * u / n_users)  # between-user spread
                for ch in CONT_CH:
                    arr[ch] = max(1e-3, base + rng.normal(0, 0.02))
            else:  # auprc
                base = _AUPRC[model]
                for ch in BIN_CH:
                    arr[ch] = float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))
            uids.append(f"u{u}")
            hls.append(100 + w)
            fls.append(24)
            arrays.append(arr.tolist())
    path.mkdir(parents=True, exist_ok=True)
    for uid in sorted(set(uids)):
        mask = [i for i, x in enumerate(uids) if x == uid]
        tbl = pa.table(
            {
                "user_id": pa.array([uids[i] for i in mask]),
                "history_length": pa.array([hls[i] for i in mask], pa.int64()),
                "forecasting_length": pa.array([fls[i] for i in mask], pa.int64()),
                metric: pa.array([arrays[i] for i in mask], pa.list_(pa.float64())),
            }
        )
        pq.write_table(tbl, path / f"{uid}.parquet")


def _make_models(tmp_path, seed=0):
    models = {}
    for i, name in enumerate(("baseline", "good", "bad")):
        root = tmp_path / name
        _write_metric_file(root / "mae", metric="mae", model=name, n_users=N_USERS, seed=seed + i)
        _write_metric_file(
            root / "auprc", metric="auprc", model=name, n_users=N_USERS, seed=seed + 10 + i
        )
        models[name] = {"path": str(root), "display_name": name}
    return models


def _bootstrap(models, **kw):
    return bootstrap_skill_rank(
        models=models,
        baseline_model="baseline",
        continuous_metrics=["mae"],
        binary_metrics=["auprc"],
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        binary_groups=[("sleep", BIN_CH)],
        **kw,
    )


def test_identity_draw_matches_point(tmp_path):
    """The no-resample (each user once) draw must reproduce the point estimate."""
    models = _make_models(tmp_path)
    metric_groups = {
        "continuous": {"metrics": ["mae"], "channel_indices": CONT_CH},
        "binary": {"metrics": ["auprc"], "channel_indices": BIN_CH},
    }
    error_df = _build_error_table(
        models=models, metric_groups=metric_groups, aggregation_unit="user"
    )
    users = sorted(set(error_df["unit_id"].astype(str)))

    # identity draw: each user exactly once
    replicas = _draw_replica_frame(users, np.arange(len(users)))
    err_b = _resample(error_df, replicas, "unit_id")
    long_b = _compute_long_skill_scores(
        error_df=err_b, models=models, baseline_model="baseline",
        clip_lower=0.01, clip_upper=100.0, min_pairs=1,
    )
    summ_b = _build_model_summary(long_df=long_b, models=models, baseline_model="baseline")

    _, point_summary, _ = compute_skill_score_tables(
        models=models, baseline_model="baseline",
        continuous_metrics=["mae"], binary_metrics=["auprc"],
        continuous_channel_indices=CONT_CH, binary_channel_indices=BIN_CH,
        clip_lower=0.01, clip_upper=100.0, min_pairs=1, aggregation_unit="user",
    )

    b = summ_b.set_index("model")
    p = point_summary.set_index("model")
    for model in ("baseline", "good", "bad"):
        for col in (
            "channel_0_score",
            "channel_1_score",
            "sleep_score",
            "channel_7_score",
            "overall_score",
        ):
            assert b.loc[model, col] == pytest.approx(p.loc[model, col], rel=1e-9, abs=1e-12)


def test_baseline_skill_brackets_zero(tmp_path):
    """Baseline vs itself -> skill mean ~ 0 and CI brackets 0."""
    models = _make_models(tmp_path)
    out = _bootstrap(models, n_boot=200, seed=1)["skill_scores"]
    base = out[out["model"] == "baseline"]
    assert not base.empty
    for _, row in base.iterrows():
        if np.isfinite(row["mean"]):
            assert abs(row["mean"]) < 0.05
            assert row["ci_lo"] <= 0.0 <= row["ci_hi"]


def test_rank_order_matches_synthetic(tmp_path):
    """Lowest-error 'good' model gets the best (lowest) mean rank."""
    models = _make_models(tmp_path)
    ranks = _bootstrap(models, n_boot=100, seed=2)["avg_rankings"]
    ch0 = ranks[(ranks["scope"] == "channel_0") & (ranks["metric"] == "mae")]
    by_model = ch0.set_index("model")["mean"].to_dict()
    assert by_model["good"] < by_model["baseline"] < by_model["bad"]


def test_output_schema_and_ci_ordering(tmp_path):
    """Both tables carry mean/se/ci_lo/ci_hi/n_boot and ci_lo<=mean<=ci_hi."""
    models = _make_models(tmp_path)
    tables = _bootstrap(models, n_boot=120, seed=3)
    for key in ("skill_scores", "avg_rankings"):
        df = tables[key]
        assert {"mean", "se", "ci_lo", "ci_hi", "n_boot"}.issubset(df.columns)
        finite = df[np.isfinite(df["mean"])]
        assert (finite["ci_lo"] <= finite["mean"] + 1e-9).all()
        assert (finite["mean"] <= finite["ci_hi"] + 1e-9).all()
        assert (finite["se"] >= 0.0).all()


def test_bootstrap_is_deterministic(tmp_path):
    """Same seed -> identical summaries (shared resample matrix is seeded)."""
    models = _make_models(tmp_path)
    a = _bootstrap(models, n_boot=80, seed=7)["skill_scores"]
    b = _bootstrap(models, n_boot=80, seed=7)["skill_scores"]
    pd.testing.assert_frame_equal(
        a.sort_values(["model", "scope"]).reset_index(drop=True),
        b.sort_values(["model", "scope"]).reset_index(drop=True),
    )


def _overall_skill_long_df(workout_channels, ratio=0.5):
    """Synthetic per-(group, metric, channel) skill rows for the overall helper."""
    rows = []
    for ch in range(0, 7):  # activity 0-4, physiology 5-6 (continuous)
        rows.append(
            dict(model="m", group="continuous", metric="mae", channel_idx=ch,
                 geometric_mean_ratio=ratio)
        )
    for ch in [7, 8, *workout_channels]:  # sleep 7-8, workout (binary)
        rows.append(
            dict(model="m", group="binary", metric="auroc", channel_idx=ch,
                 geometric_mean_ratio=ratio)
        )
    return pd.DataFrame(rows)


def test_overall_score_invariant_to_workout_channel_count():
    """Category-balanced overall_score weights each of the 4 scopes once, so adding
    workout channels (at the same ratio) leaves it unchanged — the defining property."""
    few, n_few = _aggregate_overall_category_balanced_score(_overall_skill_long_df([9, 10]), "m")
    many, n_many = _aggregate_overall_category_balanced_score(
        _overall_skill_long_df(range(9, 19)), "m"
    )
    assert n_few == n_many == 4
    assert few == pytest.approx(many)
    assert few == pytest.approx(0.5)  # all ratios 0.5 -> 1 - 0.5

    # reads only per-channel rows: a derived group row must not change it
    injected = pd.concat(
        [
            _overall_skill_long_df(range(9, 19)),
            pd.DataFrame(
                [dict(model="m", group="activity", metric="mae", channel_idx=-1,
                      geometric_mean_ratio=0.01)]
            ),
        ],
        ignore_index=True,
    )
    inj, _ = _aggregate_overall_category_balanced_score(injected, "m")
    assert inj == pytest.approx(many)


def test_overall_score_two_scope_formula():
    """overall = 1 - exp(mean over scopes of the within-scope mean log-ratio)."""
    df = pd.DataFrame(
        [
            dict(model="m", group="continuous", metric="mae", channel_idx=0,
                 geometric_mean_ratio=0.5),
            dict(model="m", group="binary", metric="auroc", channel_idx=7,
                 geometric_mean_ratio=0.8),
        ]
    )
    got, n = _aggregate_overall_category_balanced_score(df, "m")
    assert n == 2
    assert got == pytest.approx(1.0 - np.exp((np.log(0.5) + np.log(0.8)) / 2))


def test_overall_rank_invariant_and_monotonic():
    """The category-balanced overall rank weights each scope once (invariant to the
    workout channel count) and orders models by skill."""

    def user_rows(workout):
        rs = []
        for u in ("u1", "u2", "u3"):
            for ch in range(0, 7):  # continuous: lower mae is better
                for mdl, val in (("good", 0.1), ("baseline", 0.5), ("bad", 0.9)):
                    rs.append(dict(model=mdl, scope_type="continuous_channel",
                                   scope=f"channel_{ch}", scope_label="", metric="mae",
                                   metric_display="", channel_idx=ch, user_id=u,
                                   metric_value=val, n_values=1))
            for ch in (7, 8, *workout):  # binary: higher auroc is better
                for mdl, val in (("good", 0.9), ("baseline", 0.5), ("bad", 0.1)):
                    rs.append(dict(model=mdl, scope_type="binary_channel",
                                   scope=f"channel_{ch}", scope_label="", metric="auroc",
                                   metric_display="", channel_idx=ch, user_id=u,
                                   metric_value=val, n_values=1))
        return pd.DataFrame(rs)

    few = _compute_overall_category_balanced_ranks(user_rows([9, 10]))
    many = _compute_overall_category_balanced_ranks(user_rows(range(9, 19)))
    assert set(few["scope"]) == {"overall"}
    assert set(few["metric"]) == {"overall"}
    fm = few.set_index("model")["rank"].to_dict()
    mm = many.set_index("model")["rank"].to_dict()
    assert fm["good"] < fm["baseline"] < fm["bad"]
    assert all(fm[k] == pytest.approx(mm[k]) for k in fm)


def test_per_binary_channel_and_overall_in_bootstrap(tmp_path):
    """Bootstrap exposes per-binary-channel skill/rank + the category-balanced overall."""
    models = _make_models(tmp_path)
    tables = _bootstrap(models, n_boot=60, seed=4)
    skill_scopes = set(tables["skill_scores"]["scope"])
    assert "overall_score" in skill_scopes
    assert {"channel_7_score", "channel_8_score"}.issubset(skill_scopes)

    rank = tables["avg_rankings"]
    overall = rank[(rank["scope"] == "overall") & (rank["metric"] == "overall")]
    assert not overall.empty
    by_model = overall.set_index("model")["mean"].to_dict()
    assert by_model["good"] < by_model["bad"]
    # per-binary-channel rank now emitted with its binary metric
    assert not rank[(rank["scope"] == "channel_7") & (rank["metric"] == "auprc")].empty
