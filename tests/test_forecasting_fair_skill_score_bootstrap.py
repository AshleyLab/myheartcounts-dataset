"""Tests for the forecasting worst-group fairness skill score + bootstrap.

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
    _per_subgroup_skill,
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

# Test fixture has only 4 users per subgroup; relax the production default
# (50) so the fairness scope is non-empty in tests.
MIN_SUB_USERS = 1


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
        min_subgroup_users=MIN_SUB_USERS,
        **kw,
    )


def test_identity_draw_matches_point(tmp_path):
    """The no-resample (each user once) draw reproduces the deterministic point."""
    models = _make_models(tmp_path)
    demo = _demographics()
    error_df = _error_df(models)
    users = sorted(set(error_df["user_id"].astype(str)))

    point = compute_fair_skill_scores_from_errors(
        error_df,
        demo,
        baseline_method="baseline",
        clip_lower=0.01,
        clip_upper=100.0,
        min_subgroup_users=MIN_SUB_USERS,
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
    """Baseline vs itself -> per-subgroup ratio 1 -> min over subgroups = 0."""
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
    assert set(df["scope"].unique()) == {"age_group", "sex", FAIRNESS_OVERALL_SCOPE}
    finite = df[np.isfinite(df["mean"])]
    assert (finite["ci_lo"] <= finite["mean"] + 1e-9).all()
    assert (finite["mean"] <= finite["ci_hi"] + 1e-9).all()
    assert (finite["se"] >= 0.0).all()
    # The returned point table is consistent with the bootstrap key set.
    point = tables["fairness_skill_scores_point"]
    assert set(point["scope"].unique()) == {"age_group", "sex", FAIRNESS_OVERALL_SCOPE}


def test_single_subgroup_attr_dropped(tmp_path):
    """An attribute with one subgroup value (degenerate min over <2) is skipped."""
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


def test_single_subgroup_for_model_excluded_from_scope():
    """A model with data on only one subgroup of an attribute is excluded from that scope.

    Worst-group skill needs ≥ 2 subgroups for the ``min`` reduction to be a
    meaningful aggregation. A model that has data on only one subgroup of an
    attribute is dropped from that scope (and hence from the overall macro-mean).
    """
    rows = []
    # Two tasks, each with a real per-subgroup difference.
    tasks = [(0, "ch_0", 1.0, 0.5), (1, "ch_1", 1.0, 0.6)]
    for ch_idx, ch_name, e_b, e_m in tasks:
        # age_group: both baseline and model_x cover two groups (valid scope).
        for model, e in [("baseline", e_b), ("model_x", e_m)]:
            rows.append(_long_row(model, "age_group", "18-29", e, ch_idx=ch_idx, ch_name=ch_name))
            rows.append(_long_row(model, "age_group", "30-39", e * 1.1, ch_idx=ch_idx, ch_name=ch_name))
        # sex: baseline covers male+female; model_x covers ONLY male.
        rows.append(_long_row("baseline", "sex", "male", e_b, ch_idx=ch_idx, ch_name=ch_name))
        rows.append(_long_row("baseline", "sex", "female", e_b * 1.1, ch_idx=ch_idx, ch_name=ch_name))
        rows.append(_long_row("model_x", "sex", "male", e_m, ch_idx=ch_idx, ch_name=ch_name))

    out = compute_fair_skill_scores(pd.DataFrame(rows), baseline_method="baseline")

    sex_models = set(out.loc[out["scope"] == "sex", "model"].tolist())
    assert "model_x" not in sex_models, (
        "model_x has only one sex subgroup (male) and should be excluded from the 'sex' scope"
    )
    age_models = set(out.loc[out["scope"] == "age_group", "model"].tolist())
    assert "model_x" in age_models, "model_x has two age subgroups and should be present in 'age_group'"
    overall_models = set(out.loc[out["scope"] == FAIRNESS_OVERALL_SCOPE, "model"].tolist())
    assert "model_x" not in overall_models, (
        "model_x is missing the 'sex' scope and so should be excluded from the macro-mean overall"
    )


def test_worst_group_equals_min_of_per_subgroup_skills():
    """The reported S_attr is exactly min_g of the per-subgroup skill score.

    Constructs a hand-tunable per-subgroup error frame and checks the
    invariant ``S_attr == min_g (1 − exp(mean_r ln ρ_{r,g}))``.
    """
    rows = []
    # 3 tasks for age_group, 2 subgroups. baseline error = 1.0 everywhere.
    # model_x is stronger on 18-29 (ratio 0.5) than on 30-39 (ratio 0.8).
    tasks = [0, 1, 2]
    for ch_idx in tasks:
        rows.append(_long_row("baseline", "age_group", "18-29", 1.0, ch_idx=ch_idx))
        rows.append(_long_row("baseline", "age_group", "30-39", 1.0, ch_idx=ch_idx))
        rows.append(_long_row("model_x", "age_group", "18-29", 0.5, ch_idx=ch_idx))
        rows.append(_long_row("model_x", "age_group", "30-39", 0.8, ch_idx=ch_idx))

    df = pd.DataFrame(rows)
    out = compute_fair_skill_scores(df, attrs=("age_group",), baseline_method="baseline")
    s_attr = float(out.loc[out["model"] == "model_x", "fair_skill_score"].iloc[0])

    per_sub = _per_subgroup_skill(
        df[df["subgroup_attr"] == "age_group"],
        extra_keys=[],
        baseline_method="baseline",
        clip_lower=0.01,
        clip_upper=100.0,
    )
    per_sub_x = per_sub[per_sub["model"] == "model_x"].set_index("subgroup_value")["S_sub"]
    # Expected per-subgroup skills.
    assert per_sub_x.loc["18-29"] == pytest.approx(1 - 0.5, abs=1e-12)
    assert per_sub_x.loc["30-39"] == pytest.approx(1 - 0.8, abs=1e-12)
    # And the reported S_attr is the min over subgroups.
    assert s_attr == pytest.approx(min(per_sub_x.values), abs=1e-12)


def test_min_subgroup_users_drops_undersized_subgroup():
    """Subgroups below ``min_subgroup_users`` in the cohort are excluded.

    Builds 12 users with 'unknown' as a tiny 2-user bucket; with
    min_subgroup_users=4, only 18-29/30-39/40-49 contribute, and an outlier
    'unknown' subgroup must not pull the worst-group reduction.
    """
    # 4 users per age bucket (18-29, 30-39, 40-49) + 2 in 'unknown'.
    demo = {}
    for u in range(12):
        demo[f"u{u}"] = {"age_group": ["18-29", "30-39", "40-49"][u // 4], "sex": "male"}
    demo["u_outlier_a"] = {"age_group": "unknown", "sex": "male"}
    demo["u_outlier_b"] = {"age_group": "unknown", "sex": "male"}

    rows = []
    # baseline error 1.0 everywhere; model_x is 0.7 except 'unknown' where it's 1.5
    # (catastrophic). Without the floor, 'unknown' would dominate the worst-group min.
    for uid, attrs in demo.items():
        e = 1.5 if attrs["age_group"] == "unknown" else 0.7
        rows.append({
            "model": "baseline", "group": "continuous", "metric": "mae",
            "channel_idx": 0, "channel_name": "ch_0",
            "user_id": uid, "error": 1.0, "n_values": 1,
        })
        rows.append({
            "model": "model_x", "group": "continuous", "metric": "mae",
            "channel_idx": 0, "channel_name": "ch_0",
            "user_id": uid, "error": e, "n_values": 1,
        })
    err_df = pd.DataFrame(rows)

    # min_subgroup_users=4 excludes 'unknown' (n=2) but keeps the three 4-user buckets.
    out = compute_fair_skill_scores_from_errors(
        err_df,
        demo,
        attrs=("age_group",),
        baseline_method="baseline",
        min_subgroup_users=4,
    )
    s = float(out.loc[(out["model"] == "model_x") & (out["scope"] == "age_group"),
                       "fair_skill_score"].iloc[0])
    # With 'unknown' filtered out, every eligible subgroup has ratio 0.7 -> S=0.3.
    assert s == pytest.approx(0.3, abs=1e-12)

    # Sanity: with min_subgroup_users=1, 'unknown' creeps in and dominates the min.
    out_no_floor = compute_fair_skill_scores_from_errors(
        err_df,
        demo,
        attrs=("age_group",),
        baseline_method="baseline",
        min_subgroup_users=1,
    )
    s_no_floor = float(out_no_floor.loc[
        (out_no_floor["model"] == "model_x") & (out_no_floor["scope"] == "age_group"),
        "fair_skill_score"
    ].iloc[0])
    assert s_no_floor < s, (
        f"with no eligibility floor, the noisy 'unknown' subgroup must drag the worst-group "
        f"reduction down (got {s_no_floor} vs floored {s})"
    )
