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
