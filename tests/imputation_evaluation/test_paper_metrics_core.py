"""Tests for ``paper_metrics_core.compute_fair_skill_scores``.

The deterministic kernel (called from
``scripts/paper_results/compute_imputation_paper_metrics.py``) and the
per-draw bootstrap reducer (called from
``scripts/paper_results/aggregate_fairness_skill_score.py``) share the
same ``_per_attribute_skill_keyed`` helper. These tests pin that shared
behaviour: the deterministic kernel matches a single-draw run of the
bootstrap path on identical data, and well-known invariants (e.g.
baseline-vs-self skill = 0) hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.paper_metrics_core import (
    _per_attribute_skill_keyed,
    compute_fair_skill_scores,
)


def _synthetic_subgroup_errors(
    *,
    methods=("locf", "model_a"),
    scenarios=("random", "block"),
    channels=("ch_0", "ch_1"),
    base_error: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Hand-crafted long-format per-subgroup errors frame.

    ``model_a`` is uniformly better than ``locf`` on the majority subgroup
    (lower E) but has the same disparity as ``locf`` on the minority — so
    its fair-skill ratio should sit near 1 → S_attr ≈ 0 with mild noise.
    """
    if base_error is None:
        base_error = {"locf": 1.0, "model_a": 0.6}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attrs = {
        "sex": [("male", 0.0), ("female", 0.2)],   # +0.2 disparity bump
        "age_group": [("<40", 0.0), (">=40", 0.15)],
    }
    for method in methods:
        mu = base_error[method]
        for scenario in scenarios:
            for ch in channels:
                for attr, levels in attrs.items():
                    for value, bump in levels:
                        E = mu + bump + float(rng.normal(0, 1e-3))
                        rows.append({
                            "method": method,
                            "scenario": scenario,
                            "channel": ch,
                            "channel_type": "continuous",
                            "subgroup_attr": attr,
                            "subgroup_value": value,
                            "E": float(E),
                        })
    return pd.DataFrame(rows)


def test_baseline_overall_fair_skill_is_zero():
    """LOCF against itself: ratio = 1 → S_attr = 0 for every attribute."""
    errors = _synthetic_subgroup_errors()
    out = compute_fair_skill_scores(errors, baseline_method="locf")
    locf_overall = out[(out["method"] == "locf") & (out["scope"] == "overall")]
    assert not locf_overall.empty
    assert abs(float(locf_overall["fair_skill_score"].iloc[0])) < 1e-9


def test_output_schema():
    """Contract: rows are ``[method, scope, fair_skill_score, n_tasks]``."""
    errors = _synthetic_subgroup_errors()
    out = compute_fair_skill_scores(errors, baseline_method="locf")
    assert list(out.columns) == ["method", "scope", "fair_skill_score", "n_tasks"]
    # One row per attribute per method + one overall row per method.
    methods = errors["method"].unique()
    scopes = set(out["scope"].unique())
    assert scopes == {"sex", "age_group", "overall"}
    assert set(out["method"].unique()) == set(methods)


def test_deterministic_matches_single_draw_bootstrap():
    """Pin the shared-kernel contract: bootstrap with n_boot=1 on identical
    data must reproduce the deterministic fair_skill_score exactly.

    Drift between these two paths is the failure mode this refactor was
    designed to prevent — the leaderboard's point estimate and the
    bootstrap mean must come from the same arithmetic.
    """
    errors = _synthetic_subgroup_errors(seed=42)

    deterministic = compute_fair_skill_scores(errors, baseline_method="locf")

    # Re-shape into the bootstrap reducer's input: add a singleton ``draw``
    # column and dispatch through the keyed helper the same way
    # ``aggregate_fairness_skill_score._per_attribute_skill`` does.
    errors_d = errors.assign(draw=0)
    per_attr_boot: dict[str, pd.DataFrame] = {}
    for attr in ("sex", "age_group"):
        df_attr = errors_d[errors_d["subgroup_attr"] == attr]
        per_attr_boot[attr] = _per_attribute_skill_keyed(
            df_attr,
            extra_keys=["draw"],
            baseline_method="locf",
            clip_lower=1e-2,
            clip_upper=100.0,
        )

    # Per-attribute parity.
    for attr, boot_frame in per_attr_boot.items():
        det_rows = deterministic[deterministic["scope"] == attr].set_index("method")
        boot_rows = boot_frame.set_index("method")
        for method in boot_rows.index:
            assert method in det_rows.index, f"missing method {method!r} in deterministic"
            assert float(det_rows.loc[method, "fair_skill_score"]) == \
                float(boot_rows.loc[method, "S_attr"]), (
                    f"deterministic vs single-draw mismatch for method={method!r} attr={attr!r}: "
                    f"{det_rows.loc[method, 'fair_skill_score']} vs {boot_rows.loc[method, 'S_attr']}"
                )

    # Macro-average parity: per-(method, draw) S_attr from the bootstrap
    # path, averaged across attributes, must equal the deterministic
    # "overall" row exactly.
    stacked = pd.concat(
        [df.assign(attr=name) for name, df in per_attr_boot.items()],
        ignore_index=True,
    )
    boot_overall = (
        stacked.groupby("method", observed=True)["S_attr"].mean().to_dict()
    )
    det_overall = (
        deterministic[deterministic["scope"] == "overall"]
        .set_index("method")["fair_skill_score"].to_dict()
    )
    assert boot_overall.keys() == det_overall.keys()
    for method in boot_overall:
        assert det_overall[method] == boot_overall[method], (
            f"overall mismatch for method={method!r}: "
            f"det={det_overall[method]} boot={boot_overall[method]}"
        )


def test_method_missing_attribute_drops_from_overall():
    """Macro-average must drop methods that lack any attribute's rows."""
    errors = _synthetic_subgroup_errors()
    # Drop all sex rows for model_a — it now appears only in age_group.
    drop = (errors["method"] == "model_a") & (errors["subgroup_attr"] == "sex")
    errors_partial = errors[~drop].copy()

    out = compute_fair_skill_scores(errors_partial, baseline_method="locf")
    overall_methods = set(
        out.loc[out["scope"] == "overall", "method"].tolist()
    )
    # model_a missing sex → excluded from overall.
    assert "model_a" not in overall_methods
    assert "locf" in overall_methods


def test_single_subgroup_row_for_method_does_not_score_perfect_fair_skill():
    """Regression: a method with only one subgroup row for a task must NOT
    earn near-perfect fair skill.

    Old behaviour: D_j = max - min = 0 over a single row → ratio = 0 / D_b,
    clipped to ``clip_lower`` → ``S_attr ≈ 1 - clip_lower`` (≈ 0.99). With
    the fix the task is dropped from the geometric mean because the
    method ∩ baseline subgroup set has fewer than two members.
    """
    errors = _synthetic_subgroup_errors()
    # Drop one of model_a's sex subgroups on EVERY task — model_a now has
    # only the "male" row per task for the sex attribute, but the baseline
    # still has both. Under the old code, every sex task contributes
    # log(clip_lower) to model_a's mean → S_attr ≈ 0.99. Under the fix,
    # n_sub_common = 1 on every sex task → all dropped → no sex row emitted
    # for model_a, so it's excluded from the macro-average.
    drop = (
        (errors["method"] == "model_a")
        & (errors["subgroup_attr"] == "sex")
        & (errors["subgroup_value"] == "female")
    )
    errors_partial = errors[~drop].copy()

    out = compute_fair_skill_scores(errors_partial, baseline_method="locf")

    # model_a must not appear in the sex scope with a score near 1 - 1e-2.
    sex_model = out[(out["scope"] == "sex") & (out["method"] == "model_a")]
    if not sex_model.empty:
        # If a row is emitted at all, it must NOT be the bogus near-perfect
        # value produced by clipping a 0/D_b ratio.
        score = float(sex_model["fair_skill_score"].iloc[0])
        assert score < 0.5, (
            f"model_a with a single sex subgroup got bogus near-perfect "
            f"fair_skill_score={score} (expected the task to be dropped)"
        )

    # And the macro-average must drop model_a (sex contributes no tasks).
    overall_methods = set(out.loc[out["scope"] == "overall", "method"].tolist())
    assert "model_a" not in overall_methods, (
        "model_a should be excluded from overall because its sex scope is empty"
    )


def test_method_baseline_subgroup_sets_aligned():
    """When method and baseline differ on which subgroups they cover for a
    task, D_j and D_b must be computed over the **common** subgroup set.

    Construct a task where the baseline has subgroups {A, B, C} but the
    method only has {A, B}. The fair-skill ratio must use D_b = E_A - E_B
    (the common pair), not E_A - E_C, so the method does not get spuriously
    rewarded or penalised by a subgroup it never reported on.
    """
    rows = []
    # Single attribute, single task. Baseline covers three age buckets;
    # model_a covers only two. Make E_C an outlier so D_b over {A,B,C} is
    # very different from D_b over {A,B}.
    for method, e_a, e_b, e_c in [
        ("locf", 1.0, 1.2, 5.0),     # full baseline disparity = 4.0
        ("model_a", 0.6, 0.8, None),  # method only has A, B
    ]:
        for value, e in [("<40", e_a), (">=40", e_b), (">=80", e_c)]:
            if e is None:
                continue
            rows.append({
                "method": method,
                "scenario": "random",
                "channel": "ch_0",
                "channel_type": "continuous",
                "subgroup_attr": "age_group",
                "subgroup_value": value,
                "E": float(e),
            })
    errors = pd.DataFrame(rows)

    out = compute_fair_skill_scores(errors, baseline_method="locf")
    age = out[out["scope"] == "age_group"].set_index("method")

    # Expected for model_a using the aligned (A, B) subgroup set:
    #   D_j = 0.8 - 0.6 = 0.2
    #   D_b = 1.2 - 1.0 = 0.2  (NOT the full 4.0 from {A,B,C})
    #   ratio = 1.0, log(1.0) = 0  ->  S_attr = 1 - exp(0) = 0
    assert abs(float(age.loc["model_a", "fair_skill_score"]) - 0.0) < 1e-9, (
        "alignment failed: model_a's D_b should be 0.2 (common subgroups), "
        "not 4.0 (baseline-only full set)"
    )


def test_degenerate_single_subgroup_attribute_is_skipped():
    """Attributes with <2 subgroup values are skipped (max-min is degenerate)."""
    errors = _synthetic_subgroup_errors()
    # Strip all sex rows except 'male' — single value remains for sex.
    errors = errors[
        ~((errors["subgroup_attr"] == "sex") & (errors["subgroup_value"] == "female"))
    ].copy()

    out = compute_fair_skill_scores(errors, baseline_method="locf")
    assert "sex" not in set(out["scope"].unique()), \
        "sex should be skipped — only one subgroup value remains"
    # age_group still computes, and overall == age_group (single attribute macro).
    assert "age_group" in set(out["scope"].unique())
    age = out[out["scope"] == "age_group"].set_index("method")["fair_skill_score"]
    overall = out[out["scope"] == "overall"].set_index("method")["fair_skill_score"]
    for method in overall.index:
        assert overall[method] == age[method]
