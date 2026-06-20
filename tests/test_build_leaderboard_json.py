"""Tests for scripts/paper_results/build_leaderboard_json.py.

Pins the render-time-sort contract: ``_build_section`` derives display
order and ``sectionRank`` from the bootstrapped ``skill`` dict, not from
the static meta dict's insertion order. That means a CSV refresh that
reorders methods now reflows the leaderboard automatically — no human
must re-type the registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "paper_results"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import build_leaderboard_json as blj  # noqa: E402

# csv_key -> (display_name, mtype). Insertion order is deliberately NOT
# the descending-skill order, to exercise the auto-sort.
META = {
    "alpha": ("Alpha", "Statistical"),
    "beta": ("Beta", "Deep Learning"),
    "gamma": ("Gamma", "Self-Supervised"),
}


def _empty_subs() -> dict[str, dict[str, float]]:
    return {k: {} for k in META}


def test_build_section_sorts_by_skill_desc():
    """Rows come out in descending-skill order regardless of meta order."""
    skill = {"alpha": -0.10, "beta": 0.50, "gamma": 0.30}
    rows = blj._build_section(
        META,
        skill=skill,
        fair={"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        rank={"alpha": 3.0, "beta": 1.0, "gamma": 2.0},
        subs=_empty_subs(),
        submitted_on="2026-06",
    )
    methods = [r["method"] for r in rows]
    assert methods == ["Beta", "Gamma", "Alpha"]


def test_build_section_assigns_sectionRank_to_sorted_order():
    """SectionRank is 1, 2, 3 in the *sorted* order, not the meta order."""
    skill = {"alpha": 0.10, "beta": 0.80, "gamma": -0.20}
    rows = blj._build_section(
        META,
        skill=skill,
        fair={k: 0.0 for k in META},
        rank={k: float(i + 1) for i, k in enumerate(META)},
        subs=_empty_subs(),
        submitted_on="2026-06",
    )
    by_key = {r["method"]: r for r in rows}
    assert by_key["Beta"]["sectionRank"] == 1
    assert by_key["Alpha"]["sectionRank"] == 2
    assert by_key["Gamma"]["sectionRank"] == 3


def test_build_section_preserves_display_name_and_mtype():
    """The static (display_name, mtype) flows through unchanged."""
    skill = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
    rows = blj._build_section(
        META,
        skill=skill,
        fair={k: 0.0 for k in META},
        rank={k: 1.0 for k in META},
        subs=_empty_subs(),
        submitted_on="2026-06",
    )
    by_method = {r["method"]: r for r in rows}
    assert by_method["Alpha"]["mtype"] == "Statistical"
    assert by_method["Beta"]["mtype"] == "Deep Learning"
    assert by_method["Gamma"]["mtype"] == "Self-Supervised"


def test_build_section_does_not_raise_on_reordering():
    """The old SystemExit-on-order-mismatch is gone; reordering is silent."""
    # Worst-case: meta order is the *inverse* of the skill order. The old
    # implementation would have raised SystemExit; the new one just sorts.
    skill = {"alpha": 0.99, "beta": 0.50, "gamma": 0.10}  # meta order == sorted
    rows_ok = blj._build_section(
        META,
        skill=skill,
        fair={k: 0.0 for k in META},
        rank={k: 1.0 for k in META},
        subs=_empty_subs(),
        submitted_on="2026-06",
    )
    skill_inv = {"alpha": 0.10, "beta": 0.50, "gamma": 0.99}  # inverse order
    rows_inv = blj._build_section(
        META,
        skill=skill_inv,
        fair={k: 0.0 for k in META},
        rank={k: 1.0 for k in META},
        subs=_empty_subs(),
        submitted_on="2026-06",
    )
    assert [r["method"] for r in rows_ok] == ["Alpha", "Beta", "Gamma"]
    assert [r["method"] for r in rows_inv] == ["Gamma", "Beta", "Alpha"]


def test_load_overall_fairness_default(tmp_path: Path):
    """Default ``scope='overall'`` works for the fairness CSV.

    Both skill/rank and fairness CSVs use ``overall`` as the scope
    name now (the old ``overall_binary_collapsed`` was renamed). The
    semantics differ — skill/rank's ``overall`` is the 3-level B.2 over
    6 scenarios; fairness's ``overall`` is the cross-attribute macro —
    but they live in different CSVs.
    """
    csv_path = tmp_path / "fairness_skill_score_bootstrap.csv"
    csv_path.write_text(
        "method,scope,split,mean,se,ci_lo,ci_hi,n_boot\n"
        "lsm2,age_group,test,0.10,0.01,0.08,0.12,1000\n"
        "lsm2,sex,test,0.20,0.01,0.18,0.22,1000\n"
        "lsm2,overall,test,0.15,0.01,0.13,0.17,1000\n"
    )
    out = blj._load_overall(csv_path)
    assert out == {"lsm2": 0.15}


def test_load_overall_picks_overall_for_skill(tmp_path: Path):
    """Skill / rank CSVs use ``overall`` as the headline scope.

    The OVERALL_SKILL_SCOPE / OVERALL_RANK_SCOPE constants make the choice
    explicit at the import surface.
    """
    csv_path = tmp_path / "skill_scores_bootstrap.csv"
    csv_path.write_text(
        "method,scope,split,mean,se,ci_lo,ci_hi,n_boot\n"
        "lsm2,overall,test,0.55,0.01,0.53,0.57,1000\n"
        "locf,overall,test,0.00,0.00,0.00,0.00,1000\n"
    )
    out = blj._load_overall(csv_path, scope=blj.OVERALL_SKILL_SCOPE)
    assert blj.OVERALL_SKILL_SCOPE == "overall"
    assert out == {"lsm2": 0.55, "locf": 0.0}


def test_load_subgroups_reads_unified_cat_scopes(tmp_path: Path):
    """After the C3 rename, all 4 categories live under ``cat:*``.

    Per-channel binary scopes (the old ``cat:sleep`` / ``cat:workouts``
    that took the geomean over individual binary channels) were
    deleted in C2 — the unqualified label ``cat:sleep`` now means
    the binary-collapsed task.
    """
    csv_path = tmp_path / "skill_scores_bootstrap.csv"
    csv_path.write_text(
        "method,scope,split,mean,se,ci_lo,ci_hi,n_boot\n"
        "lsm2,cat:activity,test,0.30,0.01,0.28,0.32,1000\n"
        "lsm2,cat:physiology,test,0.40,0.01,0.38,0.42,1000\n"
        "lsm2,cat:sleep,test,0.50,0.01,0.48,0.52,1000\n"
        "lsm2,cat:workouts,test,0.60,0.01,0.58,0.62,1000\n"
        "lsm2,semantic,test,0.35,0.01,0.33,0.37,1000\n"
    )
    subs = blj._load_subgroups(csv_path)
    assert subs == {
        "lsm2": {
            "activity": 0.30,
            "physiology": 0.40,
            "sleep": 0.50,
            "workout": 0.60,
            "semantic": 0.35,
        }
    }


def test_subgroup_field_mapping_locked_in():
    """Lock the canonical scope→field mapping.

    Future edits to ``SUBGROUP_FIELD`` show up as test failures.
    """
    assert blj.SUBGROUP_FIELD == {
        "cat:activity": "activity",
        "cat:physiology": "physiology",
        "cat:sleep": "sleep",
        "cat:workouts": "workout",
        "semantic": "semantic",
    }


def test_full_imputation_section_with_real_registry():
    """End-to-end on the real registry with synthetic skill scores."""
    skill = {
        # DAILY_META
        "lsm2": 0.78,
        "linear": 0.27,
        "dlinear": 0.23,
        "temporal_mean": 0.08,
        "locf": 0.0,
        "brits": -0.02,
        "timesnet": -0.12,
        "temporal_mode": -0.10,
        "fedformer": -0.17,
        "mode": -0.22,
        "mean": -0.31,
        # PERSONALIZED_META
        "lsm2_weekly_sparse": 0.81,
        "personalized_temporal_mean": 0.15,
        "personalized_mean": -0.62,
        "dlinear_weekly": 0.11,
        "personalized_mode": -0.22,
    }
    zero = {k: 0.0 for k in skill}
    one = {k: 1.0 for k in skill}
    out = blj._build_imputation(
        skill=skill,
        fair=zero,
        rank=one,
        subs={},
        submitted_on="2026-06",
    )
    # Section headers preserved
    headers = [r["text"] for r in out if r.get("type") == "h1"]
    assert headers == [blj.DAILY_HEADER, blj.PERSONALIZED_HEADER]
    # Within each section, skill values are monotone non-increasing
    # Find the index of the second header to split sections
    header_idxs = [i for i, r in enumerate(out) if r.get("type") == "h1"]
    daily_rows = [r for r in out[header_idxs[0] : header_idxs[1]] if r.get("type") == "method"]
    pers_rows = [r for r in out[header_idxs[1] :] if r.get("type") == "method"]

    def _skill_pct(row: dict) -> float:
        # _fmt_pct returns "+78.0" / "0.0" / "-12.4"
        return float(row["skill"])

    daily_vals = [_skill_pct(r) for r in daily_rows]
    pers_vals = [_skill_pct(r) for r in pers_rows]
    assert daily_vals == sorted(daily_vals, reverse=True)
    assert pers_vals == sorted(pers_vals, reverse=True)
    # Specifically: timesnet should now follow temporal_mode (the swap we
    # diagnosed when refreshing the leaderboard).
    daily_methods = [r["method"] for r in daily_rows]
    assert daily_methods.index("Temporal mode") < daily_methods.index("TimesNet")
    # And personalized_mean is now at the bottom of the personalized section.
    assert pers_rows[-1]["method"] == "Pers. mean"
