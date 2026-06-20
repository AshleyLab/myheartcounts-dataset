"""Tests for the forecasting point + BCa (bias-corrected & accelerated) CIs.

Two layers:
* unit tests of the shared BCa core in ``bootstrap_skill_rank.py``
  (``_bca_interval`` / ``_jackknife_acceleration`` / ``_augment_with_bca``);
* integration tests reusing the on-disk metrics fixtures from
  ``test_forecasting_fair_skill_score_bootstrap`` (fairness, BCa on by default)
  and ``test_forecasting_bootstrap_skill_rank`` (skill/rank, BCa opt-in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Reuse the existing tiny on-disk fixtures (separate modules -> no name clash).
import test_forecasting_bootstrap_skill_rank as sr_fix
import test_forecasting_fair_skill_score_bootstrap as fair_fix

from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (
    _fairness_headline_scopes,
    _jackknife_fair_points,
)
from forecasting_evaluation.metrics.bootstrap_skill_rank import (
    _augment_with_bca,
    _bca_interval,
    _jackknife_acceleration,
)

_HEADLINE = {"overall", "activity", "sleep", "age_group", "sex"}  # present in the fixture


# --------------------------------------------------------------------------
# _bca_interval / _jackknife_acceleration — the math
# --------------------------------------------------------------------------


def test_bca_reduces_to_percentile_when_z0_a_zero():
    """(a) z0 = a = 0 -> BCa endpoints equal the percentile interval.

    Symmetric draws with the point at the median give ``prop = 0.5 -> z0 = 0``; a
    symmetric jackknife gives ``Sum d^3 = 0 -> a = 0``. The adjusted percentiles then
    collapse to alpha/2 and 1-alpha/2.
    """
    draws = np.array([-2.0, -1.0, 1.0, 2.0])
    jack = np.array([-1.0, 1.0])  # symmetric -> a = 0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo == pytest.approx(np.percentile(draws, 2.5), abs=1e-9)
    assert hi == pytest.approx(np.percentile(draws, 97.5), abs=1e-9)


def test_bca_symmetric_matches_percentile_large_sample():
    """(b) Symmetric draws + symmetric jackknife -> BCa ~ percentile at scale."""
    draws = np.concatenate([np.linspace(-5.0, -0.1, 100), np.linspace(0.1, 5.0, 100)])
    jack = np.array([-3.0, -1.0, 1.0, 3.0])  # symmetric -> a = 0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo == pytest.approx(np.percentile(draws, 2.5), abs=1e-9)
    assert hi == pytest.approx(np.percentile(draws, 97.5), abs=1e-9)


def test_bca_skewed_jackknife_shifts_endpoints():
    """A right-skewed jackknife (a < 0) shifts both endpoints below the percentile ones.

    Case (c): with the point at the median, the BCa endpoints move below the plain
    percentile endpoints — the expected direction for negative acceleration.
    """
    draws = np.concatenate([np.linspace(-10.0, -0.1, 100), np.linspace(0.1, 10.0, 100)])
    jack = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # long right tail -> a < 0
    a = _jackknife_acceleration(jack)
    assert a < 0.0
    lo, hi = _bca_interval(draws, 0.0, jack, 0.95)
    assert lo < np.percentile(draws, 2.5)
    assert hi < np.percentile(draws, 97.5)


def test_bca_all_equal_draws_returns_point():
    """(d) Degenerate (all draws equal) -> [point, point]."""
    assert _bca_interval(np.full(50, 0.5), 0.5, np.array([0.5, 0.5, 0.5]), 0.95) == (0.5, 0.5)


def test_jackknife_acceleration_hand_computed():
    """(e) Hand-computed acceleration on a tiny jackknife vector.

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
    """``point`` is set on every row; ``bca_lo``/``bca_hi`` only on in-scope rows.

    The pre-existing percentile columns are left untouched.
    """
    summary = pd.DataFrame(
        {
            "model": ["m", "m"],
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
        key_cols=["model", "scope"],
    )
    # point filled for every row (headline + non-headline)
    assert out.set_index("scope")["point"].to_dict() == {"overall": 0.12, "channel_0": 0.22}
    # bca only for the in-scope row
    ov = out[out["scope"] == "overall"].iloc[0]
    assert np.isfinite(ov["bca_lo"]) and np.isfinite(ov["bca_hi"])
    ch = out[out["scope"] == "channel_0"].iloc[0]
    assert pd.isna(ch["bca_lo"]) and pd.isna(ch["bca_hi"])
    # percentile columns untouched
    assert out["mean"].tolist() == [0.10, 0.20]
    assert out["ci_lo"].tolist() == [0.00, 0.10]


def test_augment_with_bca_empty_frame_gets_columns():
    """An empty summary still gains the three columns (stable schema)."""
    empty = pd.DataFrame(columns=["model", "scope", "mean", "se", "ci_lo", "ci_hi", "n_boot"])
    out = _augment_with_bca(
        empty,
        draws_by_key={},
        point_by_key={},
        jack_by_key={},
        scopes=frozenset(),
        ci_level=0.95,
        key_cols=["model", "scope"],
    )
    assert {"point", "bca_lo", "bca_hi"}.issubset(out.columns)
    assert out.empty


# --------------------------------------------------------------------------
# Fairness integration — BCa is ON by default
# --------------------------------------------------------------------------


def test_fairness_bca_columns_and_gating(tmp_path):
    """Fairness output carries point/bca_lo/bca_hi with BCa gated to headline scopes.

    Headline scopes get finite BCa intervals, per-channel scopes get NaN,
    ``bca_lo <= point <= bca_hi`` holds where defined, and the baseline-vs-itself
    scope collapses to ``[0, 0]``.
    """
    models = fair_fix._make_models(tmp_path)
    out = fair_fix._bootstrap(models, fair_fix._demographics(), n_boot=300, seed=11)[
        "fairness_skill_scores"
    ]
    assert {"point", "bca_lo", "bca_hi"}.issubset(out.columns)
    # point is filled for every row; bca only for headline scopes.
    assert out["point"].notna().all()
    head = out[out["scope"].isin(_HEADLINE)]
    nonhead = out[~out["scope"].isin(_HEADLINE)]
    assert head[["bca_lo", "bca_hi"]].notna().all().all()
    assert nonhead[["bca_lo", "bca_hi"]].isna().all().all()
    # BCa brackets the (reported) point.
    defined = head.dropna(subset=["bca_lo", "bca_hi", "point"])
    assert (defined["bca_lo"] <= defined["point"] + 1e-9).all()
    assert (defined["point"] <= defined["bca_hi"] + 1e-9).all()
    # Baseline vs itself -> exactly fair -> degenerate [0, 0] interval.
    base_head = out[(out["model"] == "baseline") & out["scope"].isin(_HEADLINE)]
    assert (base_head["bca_lo"].abs() < 1e-9).all()
    assert (base_head["bca_hi"].abs() < 1e-9).all()


def test_fairness_bca_preserves_percentile_columns(tmp_path):
    """``bca=True`` only *adds* columns, leaving the percentile table unchanged.

    With ``bca=False`` the legacy table is produced; for the same seed the
    percentile mean/se/CI are byte-identical whether or not BCa is enabled.
    """
    models = fair_fix._make_models(tmp_path)
    demo = fair_fix._demographics()
    without = fair_fix._bootstrap(models, demo, n_boot=120, seed=9, bca=False)[
        "fairness_skill_scores"
    ]
    with_bca = fair_fix._bootstrap(models, demo, n_boot=120, seed=9, bca=True)[
        "fairness_skill_scores"
    ]
    assert not {"point", "bca_lo", "bca_hi"}.intersection(without.columns)
    assert {"point", "bca_lo", "bca_hi"}.issubset(with_bca.columns)
    shared = ["model", "scope", "mean", "se", "ci_lo", "ci_hi", "n_boot"]
    pd.testing.assert_frame_equal(
        without[shared].sort_values(["model", "scope"]).reset_index(drop=True),
        with_bca[shared].sort_values(["model", "scope"]).reset_index(drop=True),
    )


def test_jackknife_fair_points_one_finite_value_per_user(tmp_path):
    """The fairness jackknife returns one finite value per user per headline key."""
    models = fair_fix._make_models(tmp_path)
    demo = fair_fix._demographics()
    error_df = fair_fix._error_df(models)
    users = sorted(set(error_df["user_id"].astype(str)))
    jack = _jackknife_fair_points(
        error_df,
        demo,
        users,
        baseline_model="baseline",
        attrs=("age_group", "sex"),
        clip_lower=0.01,
        clip_upper=100.0,
        scopes=_fairness_headline_scopes(("age_group", "sex")),
    )
    assert jack  # non-empty
    for key, arr in jack.items():
        assert arr.shape == (len(users),), key
        assert np.isfinite(arr).all(), key


# --------------------------------------------------------------------------
# Skill / rank integration — BCa is opt-in (default OFF)
# --------------------------------------------------------------------------


def test_skill_rank_bca_off_by_default(tmp_path):
    """Without the flag the legacy tables have no BCa columns."""
    models = sr_fix._make_models(tmp_path)
    tables = sr_fix._bootstrap(models, n_boot=60, seed=4)
    for key in ("skill_scores", "avg_rankings"):
        assert not {"point", "bca_lo", "bca_hi"}.intersection(tables[key].columns)


def test_skill_rank_bca_opt_in_well_behaved(tmp_path):
    """With the flag, headline scopes gain point + BCa; per-channel stays percentile.

    Skill is near-unbiased here, so BCa stays close to the percentile CI and brackets
    the point — confirming the machinery does not distort a well-behaved metric.
    """
    models = sr_fix._make_models(tmp_path)
    tables = sr_fix._bootstrap(models, n_boot=150, seed=21, bca_skill_rank=True)

    skill = tables["skill_scores"]
    assert {"point", "bca_lo", "bca_hi"}.issubset(skill.columns)
    ov = skill[skill["scope"] == "overall_score"].dropna(subset=["bca_lo", "bca_hi", "point"])
    assert not ov.empty
    assert (ov["bca_lo"] <= ov["point"] + 1e-9).all()
    assert (ov["point"] <= ov["bca_hi"] + 1e-9).all()
    # near-unbiased: BCa endpoints close to the percentile endpoints
    assert np.allclose(ov["bca_lo"], ov["ci_lo"], atol=0.05)
    assert np.allclose(ov["bca_hi"], ov["ci_hi"], atol=0.05)
    # per-channel skill scope keeps percentile only
    ch = skill[skill["scope"] == "channel_0_score"]
    assert ch[["bca_lo", "bca_hi"]].isna().all().all()

    rank = tables["avg_rankings"]
    assert {"point", "bca_lo", "bca_hi"}.issubset(rank.columns)
    rov = rank[(rank["scope"] == "overall") & (rank["metric"] == "overall")]
    assert rov[["point", "bca_lo", "bca_hi"]].notna().all().all()
    # per-channel rank keeps percentile only
    rch = rank[rank["scope"] == "channel_0"]
    assert rch[["bca_lo", "bca_hi"]].isna().all().all()
