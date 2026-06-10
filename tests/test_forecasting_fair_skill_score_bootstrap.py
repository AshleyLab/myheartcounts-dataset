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
    assert set(df["scope"].unique()) == {"age_group", "sex", FAIRNESS_OVERALL_SCOPE}
    finite = df[np.isfinite(df["mean"])]
    assert (finite["ci_lo"] <= finite["mean"] + 1e-9).all()
    assert (finite["mean"] <= finite["ci_hi"] + 1e-9).all()
    assert (finite["se"] >= 0.0).all()
    # The returned point table is consistent with the bootstrap key set.
    point = tables["fairness_skill_scores_point"]
    assert set(point["scope"].unique()) == {"age_group", "sex", FAIRNESS_OVERALL_SCOPE}


def test_single_subgroup_attr_dropped(tmp_path):
    """An attribute with one subgroup value (degenerate max-min) is skipped."""
    models = _make_models(tmp_path)
    out = _bootstrap(models, _demographics(constant_sex=True), n_boot=40, seed=5)
    scopes = set(out["fairness_skill_scores"]["scope"].unique())
    assert "sex" not in scopes
    assert "age_group" in scopes
    assert FAIRNESS_OVERALL_SCOPE in scopes
