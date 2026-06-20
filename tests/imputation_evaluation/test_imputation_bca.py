"""Tests for the imputation BCa (bias-corrected & accelerated) CIs.

Three layers:

* unit tests of the shared BCa core in
  ``imputation_evaluation.evaluation.bca`` (``_bca_interval`` /
  ``_jackknife_acceleration`` / ``_augment_with_bca``);
* fairness integration tests against synthetic per-user + draws fixtures —
  exercise the full LOO jackknife inside
  ``compute_fairness_skill_scores(bca=True)``;
* Phase 1 round-trip + Phase 2 skill/rank opt-in plumbing checks.

Mirrors ``tests/test_forecasting_bca.py`` for the math layer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputation_evaluation.evaluation.bca import (
    _augment_with_bca,
    _bca_interval,
    _jackknife_acceleration,
)
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    PER_USER_ERRORS_PARQUET_COLUMNS,
    read_per_user_errors_parquet,
    write_per_user_errors_parquet,
)

# Load aggregate_fairness_skill_score and aggregate_imputation_paper_metrics as
# modules even though they live under scripts/. They both define top-level
# functions we want to exercise directly without subprocess overhead.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str) -> object:
    """Import a scripts/paper_results/*.py file as a module."""
    path = _REPO_ROOT / "scripts" / "paper_results" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fairness_mod = _load_script_module("aggregate_fairness_skill_score")
paper_metrics_mod = _load_script_module("aggregate_imputation_paper_metrics")


# --------------------------------------------------------------------------
# _bca_interval / _jackknife_acceleration — the math
# --------------------------------------------------------------------------


def test_bca_reduces_to_percentile_when_z0_a_zero():
    """(a) z0 = a = 0 -> BCa endpoints equal the percentile interval.

    Symmetric draws with the point at the median give prop = 0.5 -> z0 = 0; a
    symmetric jackknife gives Sum d^3 = 0 -> a = 0. The adjusted percentiles
    then collapse to alpha/2 and 1-alpha/2.
    """
    draws = np.array([-2.0, -1.0, 1.0, 2.0])
    jack = np.array([-1.0, 1.0])  # symmetric -> a = 0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo == pytest.approx(np.percentile(draws, 2.5), abs=1e-9)
    assert hi == pytest.approx(np.percentile(draws, 97.5), abs=1e-9)


def test_bca_symmetric_matches_percentile_large_sample():
    """Symmetric draws + symmetric jackknife -> BCa ~ percentile at scale."""
    draws = np.concatenate([np.linspace(-5.0, -0.1, 100), np.linspace(0.1, 5.0, 100)])
    jack = np.array([-3.0, -1.0, 1.0, 3.0])  # symmetric -> a = 0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo == pytest.approx(np.percentile(draws, 2.5), abs=1e-9)
    assert hi == pytest.approx(np.percentile(draws, 97.5), abs=1e-9)


def test_bca_skewed_jackknife_shifts_endpoints():
    """A right-skewed jackknife (a < 0) shifts both endpoints below percentile.

    Right-skewed jackknife (a < 0) with the point at the median shifts both
    endpoints below the percentile endpoints (the expected direction).
    """
    draws = np.concatenate([np.linspace(-10.0, -0.1, 100), np.linspace(0.1, 10.0, 100)])
    jack = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # long right tail -> a < 0
    a = _jackknife_acceleration(jack)
    assert a < 0.0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo < np.percentile(draws, 2.5)
    assert hi < np.percentile(draws, 97.5)


def test_bca_all_equal_draws_returns_point():
    """Degenerate (all draws equal) -> [point, point]."""
    assert _bca_interval(np.full(50, 0.5), 0.5, np.array([0.5, 0.5, 0.5]), 0.95) == (0.5, 0.5)


def test_jackknife_acceleration_hand_computed():
    """Hand-computed acceleration on a tiny jackknife vector.

    jack = [1, 2, 3, 6] -> mean 3, d = [2, 1, 0, -3], Sum d^2 = 14, Sum d^3 = -18,
    a = -18 / (6 * 14^1.5).
    """
    assert _jackknife_acceleration(np.array([1.0, 2.0, 3.0, 6.0])) == pytest.approx(
        -18.0 / (6.0 * 14.0**1.5)
    )


def test_jackknife_acceleration_guards():
    """NaN-dropped; < 2 finite values or zero variance -> a = 0."""
    full = _jackknife_acceleration(np.array([1.0, 2.0, 3.0, 6.0]))
    assert _jackknife_acceleration(np.array([1.0, 2.0, 3.0, 6.0, np.nan])) == full
    assert _jackknife_acceleration(np.array([5.0, np.nan])) == 0.0
    assert _jackknife_acceleration(np.array([2.0, 2.0, 2.0])) == 0.0


# --------------------------------------------------------------------------
# _augment_with_bca — the column assembler + scope gating
# --------------------------------------------------------------------------


def test_augment_with_bca_fills_point_and_gates_scopes():
    """``point`` for every row; BCa bounds only for in-scope rows; others kept.

    ``point`` for every row; ``bca_lo``/``bca_hi`` only for in-scope rows; the
    percentile columns are left untouched.
    """
    summary = pd.DataFrame(
        {
            "method": ["m", "m"],
            "scope": ["overall", "channel_0"],
            "mean": [0.10, 0.20],
            "se": [0.01, 0.02],
            "ci_lo": [0.00, 0.10],
            "ci_hi": [0.20, 0.30],
            "n_boot": [100, 100],
        }
    )
    rng = np.random.default_rng(0)
    draws_by_key = {
        ("m", "overall"): rng.normal(0.1, 0.05, 200),
        ("m", "channel_0"): rng.normal(0.2, 0.05, 200),
    }
    point_by_key = {("m", "overall"): 0.12, ("m", "channel_0"): 0.22}
    jack_by_key = {
        ("m", "overall"): rng.normal(0.1, 0.01, 20),
        ("m", "channel_0"): rng.normal(0.2, 0.01, 20),
    }
    out = _augment_with_bca(
        summary,
        draws_by_key=draws_by_key,
        point_by_key=point_by_key,
        jack_by_key=jack_by_key,
        scopes=frozenset({"overall"}),
        ci_level=0.95,
        key_cols=["method", "scope"],
    )
    assert out.set_index("scope")["point"].to_dict() == {"overall": 0.12, "channel_0": 0.22}
    ov = out[out["scope"] == "overall"].iloc[0]
    assert np.isfinite(ov["bca_lo"]) and np.isfinite(ov["bca_hi"])
    ch = out[out["scope"] == "channel_0"].iloc[0]
    assert pd.isna(ch["bca_lo"]) and pd.isna(ch["bca_hi"])
    assert out["mean"].tolist() == [0.10, 0.20]
    assert out["ci_lo"].tolist() == [0.00, 0.10]


def test_augment_with_bca_empty_frame_gets_columns():
    """An empty summary still gains the three columns (stable schema)."""
    empty = pd.DataFrame(columns=["method", "scope", "mean", "se", "ci_lo", "ci_hi", "n_boot"])
    out = _augment_with_bca(
        empty,
        draws_by_key={},
        point_by_key={},
        jack_by_key={},
        scopes=frozenset(),
        ci_level=0.95,
        key_cols=["method", "scope"],
    )
    assert {"point", "bca_lo", "bca_hi"}.issubset(out.columns)
    assert out.empty


# --------------------------------------------------------------------------
# Phase 1 per-user errors Parquet round-trip
# --------------------------------------------------------------------------


def _tiny_per_user_df() -> pd.DataFrame:
    """Smallest valid per-user errors frame — covers both channel + collapsed."""
    return pd.DataFrame(
        {
            "method": ["locf", "locf", "mymodel", "mymodel"],
            "scenario": ["s1", "s1", "s1", "s1"],
            "split": ["test", "test", "test", "test"],
            "channel": ["ch_0", "cat_collapsed:sleep", "ch_0", "cat_collapsed:sleep"],
            "channel_type": ["continuous", "binary_collapsed", "continuous", "binary_collapsed"],
            "subgroup_attr": ["all", "all", "all", "all"],
            "subgroup_value": ["all", "all", "all", "all"],
            "user_id": ["u1", "u1", "u1", "u1"],
            "E_per_user": [0.5, 0.3, 0.4, 0.2],
        }
    )


def test_phase1_per_user_errors_parquet_roundtrip(tmp_path):
    """Schema + dtype contract for the per-user Parquet IO."""
    src = _tiny_per_user_df()
    path = tmp_path / "per_user_errors.parquet"
    write_per_user_errors_parquet(src, path, meta={"n_boot": 1})
    out, meta = read_per_user_errors_parquet(path)
    assert list(out.columns) == PER_USER_ERRORS_PARQUET_COLUMNS
    assert meta == {"n_boot": 1}
    # Round-trip: same values modulo categorical / float32 conversion.
    out_cmp = out.assign(**{c: out[c].astype(str) for c in PER_USER_ERRORS_PARQUET_COLUMNS[:-1]})
    pd.testing.assert_series_equal(
        out_cmp["E_per_user"].astype(np.float64).reset_index(drop=True),
        src["E_per_user"].astype(np.float64).reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


# --------------------------------------------------------------------------
# Fairness integration — BCa is ON by default
# --------------------------------------------------------------------------


def _fair_fixture(
    *,
    n_users: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired per-user + per-draw frames for the fairness BCa test.

    Two methods (``locf`` baseline + ``modelx``), two scenarios x three
    continuous channels (six tasks), two demographic attributes (``age_group``
    in {young, old}, ``sex`` in {F, M}) plus the global ``all`` cell.
    ``modelx`` E is systematically larger than ``locf`` E for the ``old`` /
    ``M`` subgroups so the max-min disparity ratio is well-defined and the
    fair skill score is positive.
    """
    rng = np.random.default_rng(seed)
    methods = ["locf", "modelx"]
    scenarios = ["s1", "s2"]
    channels = ["ch_0", "ch_1", "ch_2"]
    cells: list[tuple[str, str]] = [
        ("all", "all"),
        ("age_group", "young"),
        ("age_group", "old"),
        ("sex", "F"),
        ("sex", "M"),
    ]
    # Per-user demographics: half young/old, alternating F/M
    user_ids = [f"u{i:02d}" for i in range(n_users)]

    def _subgroup(uid: str, attr: str) -> str:
        idx = int(uid[1:])
        if attr == "age_group":
            return "young" if idx < n_users // 2 else "old"
        return "F" if idx % 2 == 0 else "M"

    # Build per_user_df: only rows where the user is in that subgroup
    per_user_rows: list[dict] = []
    for method in methods:
        for scenario in scenarios:
            for ch in channels:
                base = 1.0 if method == "locf" else 0.8
                for uid in user_ids:
                    for attr, val in cells:
                        if attr != "all" and _subgroup(uid, attr) != val:
                            continue
                        # modelx is worse for old/M; otherwise comparable to locf
                        bump = 0.0
                        if method == "modelx" and (val in ("old", "M")):
                            bump = 0.3
                        e_val = float(max(1e-6, base + bump + rng.normal(0, 0.05)))
                        per_user_rows.append(
                            {
                                "method": method,
                                "scenario": scenario,
                                "split": "test",
                                "channel": ch,
                                "channel_type": "continuous",
                                "subgroup_attr": attr,
                                "subgroup_value": val,
                                "user_id": uid,
                                "E_per_user": e_val,
                            }
                        )
    per_user_df = pd.DataFrame(per_user_rows)

    # Build draws_df by bootstrap-resampling users per (subgroup_value) cell
    n_boot = 80
    draws_rows: list[dict] = []
    baseline_mu_per_cell: dict[tuple, float] = {}
    # First pass: baseline mean per (scenario, channel, attr, val) for R
    for (scenario, ch, attr, val), grp in per_user_df[per_user_df["method"] == "locf"].groupby(
        ["scenario", "channel", "subgroup_attr", "subgroup_value"], observed=True
    ):
        baseline_mu_per_cell[(scenario, ch, attr, val)] = float(grp["E_per_user"].mean())

    for (method, scenario, ch, attr, val), grp in per_user_df.groupby(
        ["method", "scenario", "channel", "subgroup_attr", "subgroup_value"],
        observed=True,
    ):
        user_E = grp["E_per_user"].to_numpy(dtype=np.float64)
        if user_E.size == 0:
            continue
        for b in range(n_boot):
            sample_idx = rng.integers(0, user_E.size, size=user_E.size)
            mean_E = float(np.mean(user_E[sample_idx]))
            baseline_mu = baseline_mu_per_cell.get((scenario, ch, attr, val))
            r_val = float(mean_E / baseline_mu) if baseline_mu else float("nan")
            draws_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "split": "test",
                    "channel": ch,
                    "channel_type": "continuous",
                    "subgroup_attr": attr,
                    "subgroup_value": val,
                    "draw": int(b),
                    "E": mean_E,
                    "R": r_val,
                    "rank": 1.0,
                }
            )
    draws_df = pd.DataFrame(draws_rows)
    return per_user_df, draws_df


def test_fairness_bca_columns_present():
    """BCa columns are emitted, in-scope rows populated, bounds bracket the point.

    The three BCa columns are emitted, populated for in-scope rows, and BCa
    brackets the (reported) point; baseline-vs-self -> [0, 0].
    """
    per_user_df, draws_df = _fair_fixture(seed=1)
    out = fairness_mod.compute_fairness_skill_scores(
        draws_df,
        attrs=list(fairness_mod.SENSITIVE_ATTRS),
        baseline_method="locf",
        ci_level=0.95,
        bca=True,
        per_user_df=per_user_df,
    )
    assert {"point", "bca_lo", "bca_hi"}.issubset(out.columns)
    # point filled everywhere
    assert out["point"].notna().all()
    # BCa lo/hi only for the headline scopes (all 3 are headline here)
    head = out[out["scope"].isin(fairness_mod.BCA_HEADLINE_SCOPES)]
    assert not head.empty
    assert head[["bca_lo", "bca_hi"]].notna().all().all()
    # BCa brackets the (reported) point.
    defined = head.dropna(subset=["bca_lo", "bca_hi", "point"])
    assert (defined["bca_lo"] <= defined["point"] + 1e-9).all()
    assert (defined["point"] <= defined["bca_hi"] + 1e-9).all()
    # Baseline vs itself is exactly fair -> point and BCa endpoints all 0
    base = out[(out["method"] == "locf") & out["scope"].isin(fairness_mod.BCA_HEADLINE_SCOPES)]
    assert (base["point"].abs() < 1e-9).all()
    assert (base["bca_lo"].abs() < 1e-9).all()
    assert (base["bca_hi"].abs() < 1e-9).all()


def test_fairness_bca_off_matches_legacy_csv():
    """``bca=False`` keeps the legacy table; ``bca=True`` only ADDS columns.

    The percentile mean / se / CI columns are byte-identical for the same seed —
    this is the backward-compat anchor that lets us turn BCa on by default
    without invalidating any downstream consumer that reads columns by name.
    """
    per_user_df, draws_df = _fair_fixture(seed=2)
    without = fairness_mod.compute_fairness_skill_scores(
        draws_df,
        attrs=list(fairness_mod.SENSITIVE_ATTRS),
        baseline_method="locf",
        ci_level=0.95,
        bca=False,
    )
    with_bca = fairness_mod.compute_fairness_skill_scores(
        draws_df,
        attrs=list(fairness_mod.SENSITIVE_ATTRS),
        baseline_method="locf",
        ci_level=0.95,
        bca=True,
        per_user_df=per_user_df,
    )
    assert not {"point", "bca_lo", "bca_hi"}.intersection(without.columns)
    assert {"point", "bca_lo", "bca_hi"}.issubset(with_bca.columns)
    shared = ["method", "scope", "split", "n_tasks", "mean", "se", "ci_lo", "ci_hi", "n_boot"]
    pd.testing.assert_frame_equal(
        without[shared].sort_values(["method", "scope"]).reset_index(drop=True),
        with_bca[shared].sort_values(["method", "scope"]).reset_index(drop=True),
    )


def test_jackknife_fair_points_one_finite_value_per_user():
    """LOO jackknife returns one entry per user per headline key, all finite."""
    per_user_df, _ = _fair_fixture(seed=3)
    users = sorted(set(per_user_df["user_id"].astype(str)))
    jack = fairness_mod._jackknife_fair_points_from_per_user(
        per_user_df,
        attrs=list(fairness_mod.SENSITIVE_ATTRS),
        baseline_method="locf",
        clip_lower=1e-2,
        clip_upper=100.0,
        scopes=fairness_mod.BCA_HEADLINE_SCOPES,
    )
    assert jack  # non-empty
    for key, arr in jack.items():
        assert arr.shape == (len(users),), key
        assert np.isfinite(arr).all(), key


def test_fairness_bca_requires_per_user_df():
    """Calling with bca=True but no per_user_df should raise."""
    _, draws_df = _fair_fixture(seed=4)
    with pytest.raises(ValueError, match="per_user_df"):
        fairness_mod.compute_fairness_skill_scores(
            draws_df,
            attrs=list(fairness_mod.SENSITIVE_ATTRS),
            baseline_method="locf",
            bca=True,
        )


# --------------------------------------------------------------------------
# Skill / rank opt-in: default OFF, LOO not yet wired
# --------------------------------------------------------------------------


def test_skill_rank_bca_off_by_default_via_argparse():
    """``--bca-skill-rank`` defaults False — flag plumbing exists but is off.

    The CLI flag plumbing exists but is not enabled by default.
    """
    # Simulate parsing with no BCa flag.
    saved = sys.argv
    try:
        sys.argv = [
            "aggregate_imputation_paper_metrics",
            "--draws",
            "/tmp/x.parquet",
            "--output-dir",
            "/tmp/x",
        ]
        args = paper_metrics_mod._parse_args()
    finally:
        sys.argv = saved
    assert args.bca_skill_rank is False
    assert args.per_user_errors is None


def test_skill_rank_bca_opt_in_currently_stubbed(tmp_path):
    """Turning ``--bca-skill-rank`` on currently raises NotImplementedError.

    The LOO recompute for skill/rank is tracked as follow-up — see the
    error message and METRICS.md §S7.
    """
    # Build a tiny pair of parquets so the load-path validation passes.
    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        DRAWS_PARQUET_COLUMNS,
        write_draws_parquet,
    )

    draws_path = tmp_path / "draws.parquet"
    per_user_path = tmp_path / "per_user_errors.parquet"
    write_draws_parquet(
        pd.DataFrame(
            [
                {
                    "method": "locf",
                    "scenario": "s1",
                    "split": "test",
                    "channel": "ch_0",
                    "channel_type": "continuous",
                    "subgroup_attr": "all",
                    "subgroup_value": "all",
                    "draw": 0,
                    "E": 1.0,
                    "R": 1.0,
                    "rank": 1.0,
                }
            ]
        )[DRAWS_PARQUET_COLUMNS],
        draws_path,
    )
    write_per_user_errors_parquet(_tiny_per_user_df(), per_user_path)

    saved = sys.argv
    try:
        sys.argv = [
            "aggregate_imputation_paper_metrics",
            "--draws",
            str(draws_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--bca-skill-rank",
            "--per-user-errors",
            str(per_user_path),
        ]
        with pytest.raises(NotImplementedError, match="LOO jackknife"):
            paper_metrics_mod.main()
    finally:
        sys.argv = saved
