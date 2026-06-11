"""Regression tests for the ``worst_group`` sign fix under ``linear_penalty``.

Before the fix, the apply site in
:func:`imputation_evaluation.evaluation.bootstrap_skill_rank.aggregate_skill_rank_fairness`
always computed ``fairness_adjusted_<name> = S − λ·d`` regardless of the
disparity's ``higher_is_better`` flag. For ``worst_group`` (which is
registered with ``higher_is_better=True`` — higher min(S_g) means *fairer*),
that produced an inverted column: improving the worst subgroup made the
fairness-adjusted score go *down*.

The fix consults ``disparity_higher_is_better(name)`` and flips the sign of
``d`` before passing to the combine function, so that improving the
worst-group score still improves the fairness-adjusted score.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    aggregate_skill_rank_fairness,
)
from imputation_evaluation.evaluation.disparity_metrics import (
    DISPARITY_FUNCTIONS,
    FAIRNESS_COMBINE,
    disparity_higher_is_better,
)


# ---------------------------------------------------------------------------
# Helper-level: the metadata lookup must match the registry's truth.
# ---------------------------------------------------------------------------


def test_disparity_higher_is_better_matches_registry():
    """``disparity_higher_is_better`` must agree with the registered spec."""
    for name, spec in DISPARITY_FUNCTIONS.items():
        assert disparity_higher_is_better(name) is bool(spec.higher_is_better), name


def test_disparity_higher_is_better_unknown_defaults_false():
    """Unknown names default to ``False`` so legacy callers keep ``S − λ·D``."""
    assert disparity_higher_is_better("not_a_real_disparity") is False


# ---------------------------------------------------------------------------
# Apply-site math: replicate the combine to assert monotonicity directly.
# ---------------------------------------------------------------------------


def _fairness_adjusted(s_overall: float, group_scores: dict[str, float],
                       disparity_name: str, lam: float) -> float:
    """Mirror of the apply-site logic in ``aggregate_skill_rank_fairness``."""
    fn = DISPARITY_FUNCTIONS[disparity_name].fn
    combine = FAIRNESS_COMBINE["linear_penalty"]
    d = float(fn(group_scores))
    d_eff = -d if disparity_higher_is_better(disparity_name) else d
    return float(combine(s_overall, d_eff, lam))


def test_worst_group_fairness_adjusted_increases_with_min_subgroup_score():
    """Improving the worst subgroup must raise ``fairness_adjusted_worst_group``.

    Both configurations share the same overall score and same best subgroup;
    only the worst subgroup score changes. Under the (buggy) old behavior
    the configuration with the higher min would receive a *lower*
    fairness-adjusted score.
    """
    s_overall = 0.5
    lam = 1.0
    low = _fairness_adjusted(s_overall, {"a": 0.1, "b": 0.5}, "worst_group", lam)
    high = _fairness_adjusted(s_overall, {"a": 0.4, "b": 0.5}, "worst_group", lam)
    assert high > low, (
        f"worst_group fairness-adjusted should rise when min(S_g) rises; "
        f"got low={low}, high={high}"
    )


def test_max_minus_min_fairness_adjusted_decreases_with_disparity():
    """Regression guard: ``max_minus_min`` (lower-is-fairer) path unchanged."""
    s_overall = 0.5
    lam = 1.0
    fair = _fairness_adjusted(s_overall, {"a": 0.5, "b": 0.5}, "max_minus_min", lam)
    unfair = _fairness_adjusted(s_overall, {"a": 0.1, "b": 0.9}, "max_minus_min", lam)
    assert fair > unfair, (
        f"max_minus_min fairness-adjusted should fall when disparity rises; "
        f"got fair={fair}, unfair={unfair}"
    )


# ---------------------------------------------------------------------------
# Integration: run the bug all the way through aggregate_skill_rank_fairness.
# ---------------------------------------------------------------------------


def _draws_with_subgroup_gap(low_subgroup_mean: float, *, seed: int) -> pd.DataFrame:
    """Build a draws frame with two subgroups whose error gap is controlled.

    The "all" rows are the same in both configurations (so the global
    baseline + overall skill don't change). The "sex" attribute has two
    values: subgroup ``F`` with error mean = 1.0 (fixed) and subgroup ``M``
    with error mean = ``low_subgroup_mean``. The single method ``cand``
    must beat the ``locf`` baseline. We use a single channel + scenario to
    keep the synthetic input minimal.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_boot = 20
    methods_means = {"locf": 1.0, "cand": 0.5}
    # "all" rows — identical across the two configurations.
    for method, mu in methods_means.items():
        for b in range(n_boot):
            rows.append(
                {
                    "method": method,
                    "scenario": "random_noise",
                    "split": "test",
                    "channel": "ch_0",
                    "channel_type": "continuous",
                    "subgroup_attr": "all",
                    "subgroup_value": "all",
                    "draw": int(b),
                    "E": float(max(1e-6, mu + rng.normal(0, 0.02))),
                }
            )
    # Subgroup rows for the candidate method only — ``F`` fixed, ``M`` varies.
    for sg_val, sg_mean in (("F", 1.0), ("M", low_subgroup_mean)):
        for b in range(n_boot):
            rows.append(
                {
                    "method": "cand",
                    "scenario": "random_noise",
                    "split": "test",
                    "channel": "ch_0",
                    "channel_type": "continuous",
                    "subgroup_attr": "sex",
                    "subgroup_value": sg_val,
                    "draw": int(b),
                    "E": float(max(1e-6, sg_mean + rng.normal(0, 0.02))),
                }
            )
    return pd.DataFrame(rows)


def _fairness_adjusted_mean(tables: dict[str, pd.DataFrame], col: str,
                            method: str = "cand") -> float:
    summary = tables["fairness_summary"]
    sel = summary[summary["method"] == method]
    assert not sel.empty, "fairness_summary missing 'cand' method"
    return float(sel[f"{col}_mean"].iloc[0])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_aggregate_fairness_adjusted_worst_group_rises_with_min_subgroup():
    """End-to-end: improving the worst subgroup must raise
    ``fairness_adjusted_worst_group`` in the aggregator output.
    """
    # Worse subgroup error → lower S_g; we want the *better* config (M closer
    # to F) to produce a higher fairness_adjusted_worst_group.
    bad = _draws_with_subgroup_gap(low_subgroup_mean=2.0, seed=0)
    good = _draws_with_subgroup_gap(low_subgroup_mean=1.1, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bad_tables = aggregate_skill_rank_fairness(bad, baseline_method="locf")
        good_tables = aggregate_skill_rank_fairness(good, baseline_method="locf")
    bad_val = _fairness_adjusted_mean(bad_tables, "fairness_adjusted_worst_group")
    good_val = _fairness_adjusted_mean(good_tables, "fairness_adjusted_worst_group")
    assert good_val > bad_val, (
        "fairness_adjusted_worst_group should be higher when the worst "
        f"subgroup performs better; got bad={bad_val}, good={good_val}"
    )
