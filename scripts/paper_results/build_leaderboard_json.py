"""Regenerate the imputation section of the OpenMHC leaderboard JSON.

Reads the canonical seed at
`NarayanSchuetz/myheartcounts.stanford.edu:data/leaderboard/leaderboard.json`
(supplied via --current) and replaces the `imputation` array with
freshly-bootstrapped values from the Phase-2 CSVs produced by
`run_paper_pipeline.py`. The `fair_skill` field is populated from the NEW
disparity-ratio fairness skill score (macro-average across age, sex).

Section headers ("Daily (single-day) methods" /
"Personalized-context (extended-history) methods") and downstream/forecasting
arrays are preserved verbatim. `generated_at` is bumped to today.

Method order within each section is derived from the bootstrapped skill
scores at render time (descending mean); the static registry only carries
the (csv_key -> display_name, mtype) mapping. Order changes are visible
in the printed before/after diff and (more importantly) in the website PR.

Usage:
    python scripts/paper_results/build_leaderboard_json.py \
        --current  /tmp/leaderboard_canonical.json \
        --paper-dir "${SCRATCH_RUN_ROOT:-/scratch/users/$USER}/openmhc-imputation-eval/paper" \
        --output   /tmp/leaderboard_updated.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Method registry — static metadata only (display name + category). Display
# order in the JSON is computed at render time from the bootstrapped skill
# scores; the dict insertion order is irrelevant.
# ---------------------------------------------------------------------------

# csv_key -> (display_name, mtype)
DAILY_META: dict[str, tuple[str, str]] = {
    "lsm2":          ("LSM-2 (daily)",   "Self-Supervised"),
    "linear":        ("Linear",          "Statistical"),
    "dlinear":       ("DLinear",         "Deep Learning"),
    "temporal_mean": ("Temporal mean",   "Statistical"),
    "locf":          ("LOCF (baseline)", "Statistical"),
    "brits":         ("BRITS",           "Deep Learning"),
    "timesnet":      ("TimesNet",        "Deep Learning"),
    "temporal_mode": ("Temporal mode",   "Statistical"),
    "fedformer":     ("FEDformer",       "Deep Learning"),
    "mode":          ("Mode",            "Statistical"),
    "mean":          ("Mean",            "Statistical"),
}

PERSONALIZED_META: dict[str, tuple[str, str]] = {
    "lsm2_weekly_sparse":         ("LSM-2-Sparse (7-day)", "Self-Supervised"),
    "personalized_temporal_mean": ("Pers. temp. mean",     "Statistical"),
    "personalized_mean":          ("Pers. mean",           "Statistical"),
    "dlinear_weekly":             ("DLinear (7-day)",      "Deep Learning"),
    "personalized_mode":          ("Pers. mode",           "Statistical"),
}

DAILY_HEADER = "Daily (single-day) methods"
PERSONALIZED_HEADER = "Personalized-context (extended-history) methods"

SUBMITTER = "OpenMHC team"  # lowercase 't' — matches existing entries + schema doc

# CSV scope name -> JSON subgroup field name
SUBGROUP_FIELD: dict[str, str] = {
    "cat:activity":   "activity",
    "cat:physiology": "physiology",
    "cat:sleep":      "sleep",
    "cat:workouts":   "workout",   # singular in JSON
    "semantic":       "semantic",
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _load_overall(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["scope"] == "overall" and r["split"] == "test":
                out[r["method"]] = float(r["mean"])
    return out


def _load_subgroups(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            field = SUBGROUP_FIELD.get(r["scope"])
            if field is None:
                continue
            out.setdefault(r["method"], {})[field] = float(r["mean"])
    return out


# ---------------------------------------------------------------------------
# Formatting helpers (match existing JSON conventions)
# ---------------------------------------------------------------------------


def _fmt_pct(ratio: float) -> str:
    """0.8667 -> '+86.7', 0.0 -> '0.0', -0.331 -> '-33.1'."""
    pct = round(ratio * 100, 1)
    if pct == 0.0:
        return "0.0"
    return f"{pct:+.1f}"


def _fmt_rank(r: float) -> str:
    """1.09 -> '1.1' (1 decimal place, matches existing JSON)."""
    return f"{r:.1f}"


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------


def _build_section(
    meta: dict[str, tuple[str, str]],
    *,
    skill: dict[str, float],
    fair: dict[str, float],
    rank: dict[str, float],
    subs: dict[str, dict[str, float]],
    submitted_on: str,
    section_label: str = "",
) -> list[dict]:
    """Render method rows for one section, sorted by descending skill score.

    ``meta`` maps csv_key -> (display_name, mtype); ``sectionRank`` and the
    array position are derived from ``skill`` at render time so a CSV refresh
    automatically reflows the leaderboard.
    """
    ordered_keys = sorted(meta, key=lambda k: -skill[k])
    if section_label:
        pretty = ", ".join(f"{k}={skill[k]:+.3f}" for k in ordered_keys)
        print(f"[section] {section_label}: {pretty}", file=sys.stderr)

    rows: list[dict] = []
    for i, key in enumerate(ordered_keys, start=1):
        display, mtype = meta[key]
        sg = subs.get(key, {})
        rows.append({
            "type": "method",
            "method": display,
            "mtype": mtype,
            "sectionRank": i,
            "skill":      _fmt_pct(skill[key]),
            "fair_skill": _fmt_pct(fair.get(key, float("nan"))),
            "rank":       _fmt_rank(rank[key]),
            "activity":   _fmt_pct(sg.get("activity", 0.0)),
            "physiology": _fmt_pct(sg.get("physiology", 0.0)),
            "sleep":      _fmt_pct(sg.get("sleep", 0.0)),
            "workout":    _fmt_pct(sg.get("workout", 0.0)),
            "semantic":   _fmt_pct(sg.get("semantic", 0.0)),
            "submitter":  SUBMITTER,
            "submitted_on": submitted_on,
            "paper_url": "",
            "code_url": "",
        })
    return rows


def _build_imputation(
    *,
    skill: dict[str, float],
    fair: dict[str, float],
    rank: dict[str, float],
    subs: dict[str, dict[str, float]],
    submitted_on: str,
) -> list[dict]:
    out: list[dict] = []
    out.append({"type": "h1", "text": DAILY_HEADER})
    out.extend(_build_section(
        DAILY_META, skill=skill, fair=fair, rank=rank, subs=subs,
        submitted_on=submitted_on, section_label="daily",
    ))
    out.append({"type": "h1", "text": PERSONALIZED_HEADER})
    out.extend(_build_section(
        PERSONALIZED_META, skill=skill, fair=fair, rank=rank, subs=subs,
        submitted_on=submitted_on, section_label="personalized",
    ))
    return out


# ---------------------------------------------------------------------------
# Diff display
# ---------------------------------------------------------------------------


def _print_diff(old: list[dict], new: list[dict]) -> None:
    old_methods = {r["method"]: r for r in old if r.get("type") == "method"}
    new_methods = {r["method"]: r for r in new if r.get("type") == "method"}
    fields = ["skill", "fair_skill", "rank", "activity", "physiology", "sleep", "workout", "semantic"]
    print(f'{"method":<22} {"field":<12} {"old":>8}  {"new":>8}')
    print("-" * 56)
    for m in {**old_methods, **new_methods}:
        o = old_methods.get(m, {})
        n = new_methods.get(m, {})
        for f in fields:
            ov, nv = o.get(f, "—"), n.get(f, "—")
            if ov != nv:
                print(f"{m:<22} {f:<12} {ov:>8}  {nv:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--current", required=True, type=Path,
                   help="Path to the current leaderboard.json (canonical seed).")
    p.add_argument("--paper-dir", required=True, type=Path,
                   help="Directory containing the Phase-2 bootstrap CSVs.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the updated leaderboard.json.")
    p.add_argument("--submitted-on", default=date.today().strftime("%Y-%m"),
                   help="YYYY-MM stamp written to each row. Default: this month.")
    args = p.parse_args()

    current = json.loads(args.current.read_text())
    paper_dir: Path = args.paper_dir
    skill = _load_overall(paper_dir / "skill_scores_bootstrap.csv")
    fair  = _load_overall(paper_dir / "fairness_skill_score_bootstrap.csv")
    rank  = _load_overall(paper_dir / "avg_rankings_bootstrap.csv")
    subs  = _load_subgroups(paper_dir / "skill_scores_bootstrap.csv")

    missing = [k for k in (*DAILY_META, *PERSONALIZED_META) if k not in skill]
    if missing:
        raise SystemExit(f"Missing methods in skill_scores CSV: {missing}")

    new_imputation = _build_imputation(
        skill=skill, fair=fair, rank=rank, subs=subs,
        submitted_on=args.submitted_on,
    )

    updated = dict(current)
    updated["generated_at"] = date.today().isoformat()
    updated["imputation"] = new_imputation

    print(f"Diff (imputation section only):")
    print()
    _print_diff(current.get("imputation", []), new_imputation)
    print()
    args.output.write_text(json.dumps(updated, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"generated_at: {updated['generated_at']}  submitted_on: {args.submitted_on}")


if __name__ == "__main__":
    main()
