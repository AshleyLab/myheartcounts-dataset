"""Tests for the forecasting disparity-ratio fairness skill score + bootstrap.

Reuses the on-disk metrics fixture from ``test_forecasting_bootstrap_skill_rank``
(``<model>/<metric>/<user>.parquet`` with per-channel arrays), assigns synthetic
users to age/sex subgroups, and exercises both the deterministic point estimate
and the paired user-bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (
    bootstrap_fair_skill_score,
)
from forecasting_evaluation.metrics.bootstrap_skill_rank import (
    _draw_replica_frame,
    _resample,
)
from forecasting_evaluation.metrics.fair_skill_score import (
    FAIRNESS_OVERALL_SCOPE,
    _build_subgroup_error_long,
    compute_fair_skill_scores,
    compute_fair_skill_scores_from_errors,
)
from forecasting_evaluation.metrics.fairness_skill_score_summary import _build_error_table

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
                base = _AUPRC[model] * (1.0 - 0.3 * u / n_users)  # between-user spread
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


def _demographics(*, constant_sex=False):
    """Assign u0-u11 to 3 age groups and 2 sexes (or 1 sex when degenerate)."""
    demo = {}
    age_for = ["18-29", "30-39", "40-49"]
    for u in range(N_USERS):
        uid = f"u{u}"
        demo[uid] = {
            "age_group": age_for[u // 4],
            "sex": "male" if (constant_sex or u % 2 == 0) else "female",
        }
    return demo


def _error_df(models):
    return _build_error_table(
        models=models,
        continuous_metrics=["mae"],
        binary_metrics=["auprc"],
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
    )


def _bootstrap(models, demographics, **kw):
    return bootstrap_fair_skill_score(
        models=models,
        baseline_model="baseline",
        continuous_metrics=["mae"],
        binary_metrics=["auprc"],
        continuous_channel_indices=CONT_CH,
        binary_channel_indices=BIN_CH,
        demographics=demographics,
        **kw,
    )


def test_identity_draw_matches_point(tmp_path):
    """The no-resample (each user once) draw reproduces the deterministic point."""
    models = _make_models(tmp_path)
    demo = _demographics()
    error_df = _error_df(models)
    users = sorted(set(error_df["user_id"].astype(str)))

    point = compute_fair_skill_scores_from_errors(
        error_df, demo, baseline_method="baseline", clip_lower=0.01, clip_upper=100.0
    )

    replicas = _draw_replica_frame(users, np.arange(len(users)))
    demo_b = {unit: demo[orig] for orig, unit in zip(replicas["user_id"], replicas["_unit"])}
    err_b = _resample(error_df, replicas, "user_id")
    long_b = _build_subgroup_error_long(err_b, demo_b)
    fair_b = compute_fair_skill_scores(
        long_b, baseline_method="baseline", clip_lower=0.01, clip_upper=100.0
    )

    p = point.set_index(["model", "scope"])["fair_skill_score"]
    b = fair_b.set_index(["model", "scope"])["fair_skill_score"]
    assert set(p.index) == set(b.index)
    for key in p.index:
        assert b.loc[key] == pytest.approx(p.loc[key], rel=1e-9, abs=1e-12)


def test_baseline_fair_skill_is_zero(tmp_path):
    """Baseline vs itself -> disparity ratio 1 -> fair skill score exactly 0."""
    models = _make_models(tmp_path)
    out = _bootstrap(models, _demographics(), n_boot=100, seed=1)["fairness_skill_scores"]
    base = out[out["model"] == "baseline"]
    assert not base.empty
    for _, row in base.iterrows():
        assert abs(row["mean"]) < 1e-9
        assert abs(row["ci_lo"]) < 1e-9
        assert abs(row["ci_hi"]) < 1e-9


def test_determinism(tmp_path):
    """Same seed -> identical summaries (shared resample matrix is seeded)."""
    models = _make_models(tmp_path)
    demo = _demographics()
    a = _bootstrap(models, demo, n_boot=80, seed=7)["fairness_skill_scores"]
    b = _bootstrap(models, demo, n_boot=80, seed=7)["fairness_skill_scores"]
    pd.testing.assert_frame_equal(
        a.sort_values(["model", "scope"]).reset_index(drop=True),
        b.sort_values(["model", "scope"]).reset_index(drop=True),
    )


def test_schema_and_ci_ordering(tmp_path):
    """Table carries mean/se/ci_lo/ci_hi/n_boot; ci_lo<=mean<=ci_hi; scopes expected."""
    models = _make_models(tmp_path)
    tables = _bootstrap(models, _demographics(), n_boot=120, seed=3)
    df = tables["fairness_skill_scores"]
    assert {"mean", "se", "ci_lo", "ci_hi", "n_boot"}.issubset(df.columns)
    # Per-attribute + overall, plus the category-balanced per-scope and per-channel
    # rows. The fixture has continuous channels 0,1 (-> activity) and binary 7,8
    # (-> sleep), so those scopes/channels appear; physiology/workout do not.
    expected_scopes = {
        "age_group",
        "sex",
        FAIRNESS_OVERALL_SCOPE,
        "activity",
        "sleep",
        "channel_0",
        "channel_1",
        "channel_7",
        "channel_8",
    }
    assert set(df["scope"].unique()) == expected_scopes
    finite = df[np.isfinite(df["mean"])]
    assert (finite["ci_lo"] <= finite["mean"] + 1e-9).all()
    assert (finite["mean"] <= finite["ci_hi"] + 1e-9).all()
    assert (finite["se"] >= 0.0).all()
    # The returned point table is consistent with the bootstrap key set.
    point = tables["fairness_skill_scores_point"]
    assert set(point["scope"].unique()) == expected_scopes


def test_bca_columns_present_by_default(tmp_path):
    """Default output carries point + BCa for headline scopes, percentile-only for
    per-channel scopes (see test_forecasting_bca.py for the full BCa coverage)."""
    models = _make_models(tmp_path)
    out = _bootstrap(models, _demographics(), n_boot=150, seed=8)["fairness_skill_scores"]
    assert {"point", "bca_lo", "bca_hi"}.issubset(out.columns)
    assert out["point"].notna().all()
    headline = {"overall", "activity", "sleep", "age_group", "sex"}
    assert out[out["scope"].isin(headline)][["bca_lo", "bca_hi"]].notna().all().all()
    per_channel = out[out["scope"].str.startswith("channel_")]
    assert per_channel[["bca_lo", "bca_hi"]].isna().all().all()


def test_single_subgroup_attr_dropped(tmp_path):
    """An attribute with one subgroup value (degenerate max-min) is skipped."""
    models = _make_models(tmp_path)
    out = _bootstrap(models, _demographics(constant_sex=True), n_boot=40, seed=5)
    scopes = set(out["fairness_skill_scores"]["scope"].unique())
    assert "sex" not in scopes
    assert "age_group" in scopes
    assert FAIRNESS_OVERALL_SCOPE in scopes


def _long_row(model, attr, value, e, *, group="continuous", metric="mae", ch_idx=0, ch_name="ch_0"):
    """One row of the long per-subgroup error frame consumed by the metric."""
    return {
        "model": model,
        "group": group,
        "metric": metric,
        "channel_idx": ch_idx,
        "channel_name": ch_name,
        "subgroup_attr": attr,
        "subgroup_value": value,
        "E": float(e),
    }


def test_single_subgroup_row_for_model_does_not_score_perfect_fair_skill():
    """A model with only one subgroup row for a task must not score perfect fairness.

    Old behaviour: D_j = max - min = 0 over a single row -> ratio = 0 / D_b,
    clipped to ``clip_lower`` -> ``S_attr ~= 1 - clip_lower`` (~0.99). With the
    fix the task is dropped from the geometric mean because the model ∩ baseline
    subgroup set has fewer than two members.
    """
    rows = []
    # (ch_idx, ch_name, hi_baseline, hi_model): two tasks, each with a real gap.
    tasks = [(0, "ch_0", 2.0, 1.8), (1, "ch_1", 1.5, 1.4)]
    for ch_idx, ch_name, hi_b, hi_m in tasks:
        # age_group: both baseline and model_x cover two groups (valid scope).
        for model, lo, hi in [("baseline", 1.0, hi_b), ("model_x", 0.9, hi_m)]:
            rows.append(_long_row(model, "age_group", "18-29", lo, ch_idx=ch_idx, ch_name=ch_name))
            rows.append(_long_row(model, "age_group", "30-39", hi, ch_idx=ch_idx, ch_name=ch_name))
        # sex: baseline covers male+female; model_x covers ONLY male.
        rows.append(_long_row("baseline", "sex", "male", 1.0, ch_idx=ch_idx, ch_name=ch_name))
        rows.append(_long_row("baseline", "sex", "female", hi_b, ch_idx=ch_idx, ch_name=ch_name))
        rows.append(_long_row("model_x", "sex", "male", 0.9, ch_idx=ch_idx, ch_name=ch_name))

    out = compute_fair_skill_scores(pd.DataFrame(rows), baseline_method="baseline")

    # model_x's sex tasks all collapse to a single common subgroup -> dropped.
    sex_model = out[(out["scope"] == "sex") & (out["model"] == "model_x")]
    if not sex_model.empty:
        score = float(sex_model["fair_skill_score"].iloc[0])
        assert score < 0.5, (
            f"model_x with a single sex subgroup got bogus near-perfect "
            f"fair_skill_score={score} (expected the task to be dropped)"
        )

    # Sanity: model_x is meaningfully present elsewhere (its age_group scope is
    # valid), so its absence from overall is specifically the empty sex scope.
    age_models = set(out.loc[out["scope"] == "age_group", "model"].tolist())
    assert "model_x" in age_models

    overall_models = set(out.loc[out["scope"] == FAIRNESS_OVERALL_SCOPE, "model"].tolist())
    assert "model_x" not in overall_models, (
        "model_x should be excluded from overall because its sex scope is empty"
    )


def test_model_baseline_subgroup_sets_aligned():
    """D_j and D_b are computed over the common model∩baseline subgroup set per task.

    The baseline covers three age buckets {A, B, C} but the model only {A, B}.
    The fair-skill ratio must use D_b = E_A - E_B (the common pair), not the
    full E_A - E_C, so the model is neither rewarded nor penalised by a subgroup
    it never reported on.
    """
    rows = []
    # C is an outlier: D_b over {A,B,C} = 4.0, but D_b over the common {A,B} = 0.2.
    for model, e_a, e_b, e_c in [
        ("baseline", 1.0, 1.2, 5.0),
        ("model_x", 0.6, 0.8, None),  # model_x only has A, B
    ]:
        for value, e in [("18-29", e_a), ("30-39", e_b), ("40-49", e_c)]:
            if e is None:
                continue
            rows.append(_long_row(model, "age_group", value, e))

    out = compute_fair_skill_scores(pd.DataFrame(rows), baseline_method="baseline")
    age = out[out["scope"] == "age_group"].set_index("model")

    # Aligned (18-29, 30-39): D_j = 0.2, D_b = 0.2 -> ratio 1 -> S_attr = 0,
    # NOT the 4.0 baseline gap from the full {A, B, C} set.
    assert abs(float(age.loc["model_x", "fair_skill_score"]) - 0.0) < 1e-9, (
        "alignment failed: model_x's D_b should be 0.2 (common subgroups), "
        "not 4.0 (baseline-only full set)"
    )


def _fairness_rows(workout, *, attrs=("age_group", "sex")):
    """Long per-subgroup error frame: baseline gap 1.0; 'good' narrows it, 'bad'
    widens it, on every channel of every scope."""
    rows = []
    channels = list(range(0, 7)) + [7, 8] + list(workout)
    for attr in attrs:
        for ch in channels:
            group = "continuous" if ch < 7 else "binary"
            metric = "mae" if ch < 7 else "auroc"
            for model, (e1, e2) in (
                ("baseline", (1.0, 2.0)),
                ("good", (1.0, 1.2)),
                ("bad", (1.0, 4.0)),
            ):
                for value, e in (("g1", e1), ("g2", e2)):
                    rows.append(
                        _long_row(model, attr, value, e, group=group, metric=metric,
                                  ch_idx=ch, ch_name=f"c{ch}")
                    )
    return pd.DataFrame(rows)


def test_fairness_category_balanced_invariant_to_workout_count():
    """Per-attribute and overall fair skill are invariant to the workout channel
    count: each sensor scope is weighted once, so 10 workout channels == 2."""
    few = compute_fair_skill_scores(_fairness_rows([9, 10]), baseline_method="baseline")
    many = compute_fair_skill_scores(_fairness_rows(range(9, 19)), baseline_method="baseline")
    f = few.set_index(["model", "scope"])["fair_skill_score"]
    m = many.set_index(["model", "scope"])["fair_skill_score"]
    for model in ("good", "bad"):
        for scope in ("age_group", "sex", FAIRNESS_OVERALL_SCOPE):
            assert f[(model, scope)] == pytest.approx(m[(model, scope)])
    # baseline vs itself -> ratio 1 -> exactly 0; good fairer (>0), bad less fair (<0)
    assert f[("baseline", FAIRNESS_OVERALL_SCOPE)] == pytest.approx(0.0)
    assert f[("good", FAIRNESS_OVERALL_SCOPE)] > 0 > f[("bad", FAIRNESS_OVERALL_SCOPE)]


def test_fairness_per_channel_and_per_scope_rows():
    """Per-channel + per-scope fairness rows appear; a single-channel scope equals
    its channel's score (built from the same tasks)."""
    rows = []
    # activity scope here spans only channel 0; sleep spans channels 7, 8.
    tasks = [(0, "continuous", "mae"), (7, "binary", "auroc"), (8, "binary", "auroc")]
    for attr in ("age_group", "sex"):
        for ch, group, metric in tasks:
            for model, (e1, e2) in (("baseline", (1.0, 2.0)), ("good", (1.0, 1.3))):
                for value, e in (("g1", e1), ("g2", e2)):
                    rows.append(
                        _long_row(model, attr, value, e, group=group, metric=metric,
                                  ch_idx=ch, ch_name=f"c{ch}")
                    )
    out = compute_fair_skill_scores(pd.DataFrame(rows), baseline_method="baseline")
    scopes = set(out["scope"])
    assert {"activity", "sleep"}.issubset(scopes)  # per-scope rows
    assert {"channel_0", "channel_7", "channel_8"}.issubset(scopes)  # per-channel rows
    # activity has a single channel (0), so its scope score == the channel_0 score.
    good = out[out["model"] == "good"].set_index("scope")["fair_skill_score"]
    assert good["activity"] == pytest.approx(good["channel_0"])
