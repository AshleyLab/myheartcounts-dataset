"""Tests for the fairness skill-score BCa interval + leave-one-user-out jackknife.

Covers the BCa math (it must reduce to the percentile interval when the draws are
symmetric and the jackknife is symmetric, and fall back to percentile on
degenerate input) and the central correctness property: the jackknife's
full-cohort point reproduces the bootstrap POINT_DRAW value, so the deterministic
score is unchanged and only the interval gains the correction.
"""

import numpy as np
import pytest

from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    POINT_DRAW,
    _attr_disparity_ratio_skill,
    _bca_interval,
    aggregate_skill_rank_fairness,
    compute_per_draw_errors,
    jackknife_fairness_skill,
    jackknife_skill_rank,
)

ATTRS = ("age_group", "sex")
CLIP_LO, CLIP_HI = 1e-2, 100.0
# Synthetic tasks aren't in the real TASK_DOMAIN_MAP; map them to two domains so the
# skill/rank macro (per-domain + Overall) is well defined in tests.
DOMMAP = {"t_bin": "D1", "t_reg": "D2"}


# --------------------------- BCa math ---------------------------


def test_bca_reduces_to_percentile_when_symmetric():
    """z0 = a = 0 (draws symmetric about the point, symmetric jackknife) -> percentile."""
    draws = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float)
    jack = np.array([-2, -1, 0, 1, 2], dtype=float)  # symmetric -> acceleration 0
    lo, hi = _bca_interval(draws, point=0.0, jack=jack, ci_level=0.95)
    assert lo == pytest.approx(float(np.percentile(draws, 2.5)))
    assert hi == pytest.approx(float(np.percentile(draws, 97.5)))


def test_bca_skews_interval_for_right_skewed_draws():
    """A right-skewed bootstrap with the point above the median shifts the CI up."""
    rng = np.random.default_rng(0)
    draws = rng.gamma(shape=2.0, scale=1.0, size=2000)
    point = float(np.percentile(draws, 70))  # point sits above the median
    jack = rng.gamma(shape=2.0, scale=1.0, size=200)
    lo, hi = _bca_interval(draws, point=point, jack=jack, ci_level=0.95)
    plo = float(np.percentile(draws, 2.5))
    phi = float(np.percentile(draws, 97.5))
    # Bias correction (z0 > 0 here) pushes both endpoints above the percentile ones.
    assert lo > plo
    assert hi > phi


def test_bca_falls_back_on_degenerate_input():
    """Empty draws -> (nan, nan); zero-spread draws -> [point, point]."""
    jack = np.array([0.1, 0.2, 0.3])
    assert _bca_interval(np.array([]), 0.5, jack, 0.95) == (
        pytest.approx(float("nan"), nan_ok=True),
        pytest.approx(float("nan"), nan_ok=True),
    )
    # all draws equal -> degenerate spread -> [point, point]
    assert _bca_interval(np.full(50, 0.3), 0.42, jack, 0.95) == (0.42, 0.42)


# --------------------------- jackknife identity ---------------------------


def _synthetic_cohort():
    """2 methods x 2 tasks (binary + regression), 40 users, 2x2 balanced subgroups.

    The baseline's prediction quality varies by subgroup (better for male / young),
    so every attribute carries a non-zero baseline disparity D_base > 0 and the
    disparity-ratio score is actually defined. Subgroups stay class-balanced before
    and after dropping any single user, keeping all metrics finite.
    """
    rng = np.random.default_rng(0)
    n = 40
    uids = np.array([f"u{i}" for i in range(n)])
    sex = np.array(["male" if i < 20 else "female" for i in range(n)])
    age = np.array(["young" if (i // 10) % 2 == 0 else "old" for i in range(n)])
    y_bin = rng.integers(0, 2, n).astype(float)
    y_reg = rng.normal(0.0, 1.0, n)

    def proba(corr):
        # corr in [0, 1]: 1 -> proba tracks the label (high AUPRC), 0 -> noise.
        return np.clip(corr * y_bin + (1.0 - corr) * rng.uniform(0.0, 1.0, n), 0.0, 1.0)

    def reg_pred(corr):
        return corr * y_reg + (1.0 - corr) * rng.normal(0.0, 1.0, n)

    corr_base = np.where(sex == "male", 0.85, 0.45) - np.where(age == "old", 0.2, 0.0)
    corr_mae = np.full(n, 0.7)
    pb_base, pb_mae = proba(corr_base), proba(corr_mae)
    aligned = {
        "linear": {
            "t_bin": {
                "uids": uids,
                "y_true": y_bin,
                "y_pred": (pb_base > 0.5).astype(float),
                "y_proba": pb_base,
                "task_type": "binary",
            },
            "t_reg": {
                "uids": uids,
                "y_true": y_reg,
                "y_pred": reg_pred(corr_base),
                "y_proba": np.zeros(n),
                "task_type": "regression",
            },
        },
        "mae": {
            "t_bin": {
                "uids": uids,
                "y_true": y_bin,
                "y_pred": (pb_mae > 0.5).astype(float),
                "y_proba": pb_mae,
                "task_type": "binary",
            },
            "t_reg": {
                "uids": uids,
                "y_true": y_reg,
                "y_pred": reg_pred(corr_mae),
                "y_proba": np.zeros(n),
                "task_type": "regression",
            },
        },
    }
    subgroup_map = {f"u{i}": {"sex": sex[i], "age_group": age[i]} for i in range(n)}
    return aligned, subgroup_map, uids


def _draws_path_point(draws, methods, base):
    """Replicate the aggregator's POINT_DRAW reduction from the draws frame."""
    sub = draws[draws["subgroup_attr"].isin(ATTRS)]
    out, per_attr = {}, {m: {} for m in methods}
    for attr in ATTRS:
        g = sub[(sub["subgroup_attr"] == attr) & (sub["draw"] == POINT_DRAW)]
        for m, s in _attr_disparity_ratio_skill(g, methods, base, CLIP_LO, CLIP_HI).items():
            out[(m, attr)] = s
            per_attr[m][attr] = s
    for m in methods:
        vals = [per_attr[m][a] for a in ATTRS if a in per_attr[m]]
        if vals:
            out[(m, "overall")] = float(np.mean(vals))
    return out


def test_jackknife_point_matches_draws_point():
    """The jackknife's full-cohort point equals the bootstrap POINT_DRAW value."""
    aligned, subgroup_map, uids = _synthetic_cohort()
    base = "linear"
    draws = compute_per_draw_errors(
        aligned, n_bootstrap=3, seed=0, subgroup_map=subgroup_map, subgroup_attributes=list(ATTRS)
    )
    methods = sorted(draws["method"].unique())
    expected = _draws_path_point(draws, methods, base)

    jack, point = jackknife_fairness_skill(
        aligned, subgroup_map, ATTRS, base, clip_lower=CLIP_LO, clip_upper=CLIP_HI
    )

    assert set(point) == set(expected)
    assert expected  # sanity: the reduction actually produced scores
    for key, value in expected.items():
        assert point[key] == pytest.approx(value, abs=1e-9), key
    # one jackknife replicate per distinct user, for every scored (method, scope)
    n_users = len({u for t in aligned["linear"] for u in aligned["linear"][t]["uids"]})
    for arr in jack.values():
        assert arr.shape == (n_users,)


# --------------------------- skill / rank jackknife ---------------------------


def test_jackknife_skill_rank_point_matches_draws_point():
    """Skill/rank jackknife full-cohort points equal the draws POINT_DRAW values."""
    aligned, subgroup_map, _ = _synthetic_cohort()
    draws = compute_per_draw_errors(
        aligned, n_bootstrap=4, seed=0, subgroup_map=subgroup_map,
        subgroup_attributes=list(ATTRS), domain_map=DOMMAP,
    )
    tables = aggregate_skill_rank_fairness(
        draws, baseline="linear", clip_lower=CLIP_LO, clip_upper=CLIP_HI, domain_map=DOMMAP
    )
    skill_jack, rank_jack, skill_pt, rank_pt = jackknife_skill_rank(
        aligned, "linear", clip_lower=CLIP_LO, clip_upper=CLIP_HI, domain_map=DOMMAP
    )
    assert skill_pt and rank_pt  # sanity: the macro is actually defined
    sk = tables["skill_scores"].set_index(["method", "scope"])["point"]
    for key, value in skill_pt.items():
        assert sk.loc[key] == pytest.approx(value, abs=1e-9), ("skill", key)
    rk = tables["avg_rankings"].set_index(["method", "scope"])["point"]
    for key, value in rank_pt.items():
        assert rk.loc[key] == pytest.approx(value, abs=1e-9), ("rank", key)
    n_users = len({u for t in aligned["linear"] for u in aligned["linear"][t]["uids"]})
    for arr in list(skill_jack.values()) + list(rank_jack.values()):
        assert arr.shape == (n_users,)


def test_skill_rank_bca_columns_only_when_enabled():
    """BCa columns appear only with bca_skill_rank=True; shared columns are unchanged."""
    import pandas as pd

    aligned, subgroup_map, _ = _synthetic_cohort()
    draws = compute_per_draw_errors(
        aligned, n_bootstrap=8, seed=0, subgroup_map=subgroup_map,
        subgroup_attributes=list(ATTRS), domain_map=DOMMAP,
    )
    common = dict(baseline="linear", clip_lower=CLIP_LO, clip_upper=CLIP_HI, domain_map=DOMMAP)
    off = aggregate_skill_rank_fairness(draws, **common)
    on = aggregate_skill_rank_fairness(draws, **common, aligned=aligned, bca_skill_rank=True)

    for tbl in ("skill_scores", "avg_rankings"):
        assert "bca_lo" not in off[tbl].columns
        assert {"bca_lo", "bca_hi"}.issubset(on[tbl].columns)
        # turning BCa on must not perturb the percentile/point columns
        shared = [c for c in off[tbl].columns]
        pd.testing.assert_frame_equal(off[tbl][shared], on[tbl][shared])
