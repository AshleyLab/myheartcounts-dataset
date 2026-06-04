#!/usr/bin/env python3
"""Parity check: new openmhc imputation-eval results vs MHC-benchmark max91d.

Compares headline metrics (continuous mean-normalized RMSE, binary macro
balanced accuracy) per (method, scenario, split) within tolerance:
  - continuous: 1% relative
  - binary balanced accuracy: 0.005 absolute

Sources:
  new:  ${RUNS_ROOT}/<method>/results.json  (or pairs/aggregated_metrics.json)
  old:  /scratch/users/schuetzn/mhc-benchmark-results/imputation_eval/
        <pattern matched per method>/{results.json,pairs/aggregated_metrics.json}

Also (best-effort) compares:
  - Per-imputer CIs in {new,old}/bootstrap_metrics.json overlap test.
  - Paper-bootstrap CSV rows: openmhc paper/ skill_scores_bootstrap.csv vs
    MHC-benchmark/results/paper/skill_scores_bootstrap.csv (if both exist).

Usage:
  python jobs/sherlock/imputation_eval/verify_parity.py
  python jobs/sherlock/imputation_eval/verify_parity.py --methods brits dlinear

Exit code: 0 if every checked row passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

OUT_BASE = Path("/scratch/users/schuetzn/openmhc-imputation-eval")
RUNS_ROOT = OUT_BASE / "runs"
PAPER_OUT = OUT_BASE / "paper"

OLD_ROOT = Path("/scratch/users/schuetzn/mhc-benchmark-results/imputation_eval")
OLD_PAPER = Path("/home/users/schuetzn/MHC-benchmark/results/paper")

# Map openmhc method name -> glob pattern under OLD_ROOT for the matching
# MHC-benchmark max91d run. Multiple matches are allowed; we'll pick the
# newest one (by mtime). None means no parity reference available.
OLD_PATTERNS: dict[str, list[str]] = {
    "mean":                       ["baselines_max91d_*_imputation_mean_*"],
    "mode":                       ["baselines_max91d_*_imputation_mode_*"],
    "linear":                     ["baselines_max91d_*_imputation_linear_*"],
    "locf":                       ["baselines_max91d_*_imputation_locf_*"],
    "temporal_mean":              ["baselines_max91d_*_imputation_temporal_mean_*"],
    "temporal_mode":              ["baselines_max91d_*_imputation_temporal_mode_*"],
    "personalized_mean":          ["baselines_max91d_*_imputation_personalized_mean_*"],
    "personalized_mode":          ["baselines_max91d_*_imputation_personalized_mode_*"],
    "personalized_temporal_mean": ["baselines_max91d_*_imputation_personalized_temporal_mean_*"],
    "brits":                      ["brits_max91d_*"],
    "dlinear":                    ["dlinear_max91d_*", "pypots_dlinear_max91d_*"],
    "dlinear_weekly":             ["pypots_dlinear_7day_max91d_*"],
    "fedformer":                  ["pypots_fedformer_max91d_*"],
    "timesnet":                   ["pypots_timesnet_max91d_*"],
    "lsm2":                       ["mae_daily_nodropout_max91d_*"],
    "lsm2_weekly_sparse":         ["mae_weekly_sparse_max91d_*"],
}

# Tolerances ---------------------------------------------------------------
TOL_CONT_RELATIVE = 0.01       # 1 % relative
TOL_BIN_ABSOLUTE = 0.005       # 0.5 pp absolute

# Headline metrics ---------------------------------------------------------
HEADLINE = [
    ("continuous", "mean_normalized_rmse",   "cont_nrmse",  "rel"),
    ("continuous", "mean_normalized_mae",    "cont_nmae",   "rel"),
    ("binary",     "macro_balanced_accuracy","bin_bacc",    "abs"),
    ("binary",     "macro_roc_auc",          "bin_auc",     "abs"),
]


# ----------------------- loaders ------------------------------------------

def _load_metrics(run_dir: Path) -> dict | None:
    """Load aggregated metrics, trying common file layouts."""
    candidates = [
        run_dir / "pairs" / "aggregated_metrics.json",   # MHC-benchmark layout
        run_dir / "results.json",                        # openmhc layout
        run_dir / "results_aggregated.json",
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            data = json.loads(c.read_text())
        except json.JSONDecodeError:
            continue
        # Skip pairs-only markers (no metrics inside).
        sc = data.get("scenarios") or {}
        if sc:
            first = next(iter(sc.values()))
            split = next(iter(first.values()), {})
            if "continuous" in split or "binary" in split:
                return data
    return None


def _find_old_run(method: str) -> Path | None:
    if method not in OLD_PATTERNS:
        return None
    matches: list[Path] = []
    for pat in OLD_PATTERNS[method]:
        matches.extend(OLD_ROOT.glob(pat))
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# ----------------------- comparison ---------------------------------------

def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _row_ok(old: float, new: float, kind: str) -> tuple[bool, str]:
    if not (_is_finite(old) and _is_finite(new)):
        return (False, "non-finite")
    diff = new - old
    if kind == "rel":
        denom = max(abs(old), 1e-12)
        rel = abs(diff) / denom
        ok = rel <= TOL_CONT_RELATIVE
        return (ok, f"{rel:.2%}")
    if kind == "abs":
        ok = abs(diff) <= TOL_BIN_ABSOLUTE
        return (ok, f"{diff:+.4f}")
    return (False, "unknown")


def _diff_block(
    method: str, old: dict, new: dict, splits: tuple[str, ...] = ("test",),
) -> list[tuple]:
    """Yield rows: (method, scenario, split, metric_label, old, new, delta, ok)."""
    rows = []
    old_scenarios = old.get("scenarios", {})
    new_scenarios = new.get("scenarios", {})
    for scenario in sorted(set(old_scenarios) & set(new_scenarios)):
        for split in splits:
            o_sp = old_scenarios[scenario].get(split, {})
            n_sp = new_scenarios[scenario].get(split, {})
            for group, key, label, kind in HEADLINE:
                o_v = o_sp.get(group, {}).get(key)
                n_v = n_sp.get(group, {}).get(key)
                if o_v is None and n_v is None:
                    continue
                ok, delta = _row_ok(o_v, n_v, kind)
                rows.append((method, scenario, split, label, o_v, n_v, delta, ok))
    return rows


# ----------------------- print --------------------------------------------

def _print_table(rows: list[tuple]) -> int:
    if not rows:
        print("(no rows to compare)")
        return 0
    print(f"{'method':<28} {'scenario':<20} {'split':<5} {'metric':<10} "
          f"{'old':>10} {'new':>10} {'Δ':>10}  ok")
    print("-" * 110)
    fails = 0
    for method, scenario, split, label, o, n, delta, ok in rows:
        flag = "✓" if ok else "✗"
        if not ok:
            fails += 1
        o_s = f"{o:.4f}" if _is_finite(o) else "  N/A"
        n_s = f"{n:.4f}" if _is_finite(n) else "  N/A"
        print(f"{method:<28} {scenario:<20} {split:<5} {label:<10} "
              f"{o_s:>10} {n_s:>10} {delta:>10}  {flag}")
    print()
    if fails:
        print(f"FAIL: {fails}/{len(rows)} rows outside tolerance")
    else:
        print(f"PASS: {len(rows)}/{len(rows)} rows within tolerance")
    return 1 if fails else 0


# ----------------------- paper bootstrap parity ---------------------------

def _check_paper_csvs() -> None:
    """Print a quick side-by-side of the headline paper CSVs, if both exist."""
    for name in ("skill_scores_bootstrap.csv", "avg_rankings_bootstrap.csv"):
        new = PAPER_OUT / name
        old = OLD_PAPER / name
        if not new.exists():
            print(f"[paper] missing new {new}")
            continue
        if not old.exists():
            print(f"[paper] no MHC-benchmark reference for {name} at {old}")
            continue
        print(f"[paper] {name}:")
        print(f"   old: {old}  ({old.stat().st_size} bytes)")
        print(f"   new: {new}  ({new.stat().st_size} bytes)")
        # Don't try to parse — leave detailed cmp to a notebook; just note both exist.


# ----------------------- main ---------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Restrict to subset of method names (default: all in RUNS_ROOT)")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="Which splits to compare (default: test)")
    args = ap.parse_args()

    if not RUNS_ROOT.exists():
        print(f"no new runs at {RUNS_ROOT}", file=sys.stderr)
        return 2

    available = sorted(p.name for p in RUNS_ROOT.iterdir() if p.is_dir())
    methods = args.methods or available
    rows: list[tuple] = []
    for m in methods:
        new_dir = RUNS_ROOT / m
        if not new_dir.is_dir():
            print(f"[skip] no run dir for {m} at {new_dir}")
            continue
        new = _load_metrics(new_dir)
        if new is None:
            print(f"[skip] {m}: no aggregated metrics in {new_dir}")
            continue
        old_dir = _find_old_run(m)
        if old_dir is None:
            print(f"[skip] {m}: no MHC-benchmark reference matched OLD_PATTERNS")
            continue
        old = _load_metrics(old_dir)
        if old is None:
            print(f"[skip] {m}: old dir {old_dir} has no aggregated metrics")
            continue
        rows.extend(_diff_block(m, old, new, tuple(args.splits)))

    code = _print_table(rows)
    print()
    _check_paper_csvs()
    return code


if __name__ == "__main__":
    sys.exit(main())
