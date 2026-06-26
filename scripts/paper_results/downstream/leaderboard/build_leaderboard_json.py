"""Regenerate the downstream (prediction-task) section of the OpenMHC leaderboard JSON.

Reads a current ``leaderboard.json`` seed (``--current``), replaces the
``downstream`` array with values from the phase-2 sidecar CSVs
(``skill_scores_bootstrap.csv``, ``fairness_skill_score_bootstrap.csv``,
``avg_rankings_bootstrap.csv``), and writes the result. Other track arrays
(``imputation``, ``forecasting``) are preserved verbatim; ``generated_at`` is
bumped to today.

NOTE — provisional bits to confirm against the live leaderboard schema before
publishing: the method registry :data:`METHODS` (csv key → display name → type),
the per-domain JSON field names :data:`DOMAIN_FIELD`, and the leaderboard seed.

Usage::

    python scripts/paper_results/downstream/leaderboard/build_leaderboard_json.py \
        --current   /tmp/leaderboard.json \
        --paper-dir results/paper \
        --output    /tmp/leaderboard_updated.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

# (csv_key, display_name, mtype) — static metadata only. The display order and
# sectionRank are computed at render time from the bootstrapped skill scores
# (descending), so a CSV refresh reflows the leaderboard automatically; the list
# order here is irrelevant.
METHODS: list[tuple[str, str, str]] = [
    ("lsm2", "LSM-2", "Self-Supervised"),
    ("xgboost", "XGBoost", "Statistical"),
    ("multirocket", "MultiRocket", "Convolutional"),
    ("wbm", "WBM", "Self-Supervised"),
    ("linear", "Linear (baseline)", "Statistical"),
    ("gru_d", "GRU-D", "Deep Learning"),
    ("chronos2", "Chronos-2", "Foundation"),
    ("toto", "Toto", "Foundation"),
]

SUBMITTER = "OpenMHC team"

# skill_scores scope (domain) -> JSON per-domain field name.
DOMAIN_FIELD: dict[str, str] = {
    "Demographics": "demographics",
    "Medical conditions": "medical",
    "Body metrics and biomarkers": "biomarkers",
    "Mental well-being": "mental",
    "Wearable physiology": "wearable",
    "Sleep and lifestyle": "sleep",
}


def _load_scope(path: Path, scope: str, value_col: str = "point") -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["scope"] == scope:
                out[r["method"]] = float(r[value_col])
    return out


def _load_domains(path: Path, value_col: str = "point") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            field = DOMAIN_FIELD.get(r["scope"])
            if field is None:
                continue
            out.setdefault(r["method"], {})[field] = float(r[value_col])
    return out


def _fmt_pct(x: float) -> str:
    """0.8667 -> '+86.7', 0.0 -> '0.0', -0.331 -> '-33.1'."""
    pct = round(x * 100, 1)
    return "0.0" if pct == 0.0 else f"{pct:+.1f}"


def _fmt_rank(r: float) -> str:
    return f"{r:.1f}"


def _build_section(
    methods: list[tuple[str, str, str]],
    *,
    skill: dict[str, float],
    fair: dict[str, float],
    rank: dict[str, float],
    doms: dict[str, dict[str, float]],
    submitted_on: str,
) -> list[dict]:
    """Render method rows, sorted by descending skill score.

    ``methods`` carries the (csv_key, display_name, mtype) mapping only; the
    display order and ``sectionRank`` are derived from ``skill`` at render time,
    so a CSV refresh that reorders methods reflows the leaderboard automatically.
    """
    ordered = sorted(methods, key=lambda m: -skill[m[0]])
    rows: list[dict] = []
    for i, (key, display, mtype) in enumerate(ordered, start=1):
        dd = doms.get(key, {})
        row = {
            "type": "method",
            "method": display,
            "mtype": mtype,
            "sectionRank": i,
            "skill": _fmt_pct(skill[key]),
            "fair_skill": _fmt_pct(fair.get(key, float("nan"))),
            "rank": _fmt_rank(rank[key]),
            "submitter": SUBMITTER,
            "submitted_on": submitted_on,
            "paper_url": "",
            "code_url": "",
        }
        for field in DOMAIN_FIELD.values():
            row[field] = _fmt_pct(dd.get(field, 0.0))
        rows.append(row)
    return rows


def main() -> None:
    """Replace the ``downstream`` array of the leaderboard JSON from the sidecar CSVs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--current", required=True, type=Path, help="Current leaderboard.json seed.")
    p.add_argument("--paper-dir", required=True, type=Path, help="Dir with the phase-2 CSVs.")
    p.add_argument("--output", required=True, type=Path, help="Where to write the updated JSON.")
    p.add_argument("--submitted-on", default=date.today().strftime("%Y-%m"))
    args = p.parse_args()

    current = json.loads(args.current.read_text())
    paper_dir: Path = args.paper_dir
    # Match imputation/forecasting: skill score + average rank report the bootstrap
    # ``mean``; the BCa-corrected fairness skill score reports the deterministic ``point``.
    skill = _load_scope(paper_dir / "skill_scores_bootstrap.csv", "Overall", "mean")
    rank = _load_scope(paper_dir / "avg_rankings_bootstrap.csv", "Overall", "mean")
    fair = _load_scope(paper_dir / "fairness_skill_score_bootstrap.csv", "overall", "point")
    doms = _load_domains(paper_dir / "skill_scores_bootstrap.csv", "mean")

    missing = [k for k, _, _ in METHODS if k not in skill]
    if missing:
        raise SystemExit(f"Missing methods in skill_scores CSV: {missing}")

    section = _build_section(
        METHODS,
        skill=skill,
        fair=fair,
        rank=rank,
        doms=doms,
        submitted_on=args.submitted_on,
    )
    updated = dict(current)
    updated["generated_at"] = date.today().isoformat()
    updated["downstream"] = section
    args.output.write_text(json.dumps(updated, indent=2) + "\n")
    print(f"Wrote {args.output} — downstream section: {len(section)} methods")


if __name__ == "__main__":
    main()
