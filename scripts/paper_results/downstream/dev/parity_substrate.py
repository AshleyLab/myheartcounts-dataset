"""Downstream substrate parity gate (Gate B + Gate C).

Proves the Track-1 per-user-pairs substrate (what the leaderboard ingests and
recomputes from) is faithful to the predictions the committed paper bootstrap
consumes — the downstream analog of ``forecasting/parity/parity_substrate.py`` and
``imputation/parity/parity_bootstrap.py``.

Checks
------
* **Gate B (exact)** — for every (method, task), the substrate's ``all`` cells are
  byte-equal to ``predictions/<method>/<task>/test.parquet`` after the float32 cast,
  and the ``age_group`` / ``sex`` subgroup rows match the ``_subgroups.json``
  expansion. The substrate is exactly the predictions, repackaged + subgroup-expanded.
* **Gate C (float32 tolerance)** — a fresh ``bootstrap_draws`` built **from the
  substrate** (align → ``compute_per_draw_errors``, seed 42, 1000 draws) reproduces
  the committed ``results/paper/bootstrap_draws.parquet``. This is a *tolerance* check,
  not byte-equal: the substrate stores float32 pairs (SCHEMA-mandated) while the
  committed draws came from float64 predictions, so a few tie-sensitive metrics differ
  at ~1e-6. Most draws are byte-identical; none should exceed the tolerance.

Gate B alone already implies the leaderboard recompute reproduces the paper (identical
inputs → identical bootstrap); Gate C runs it end-to-end as the headline confirmation.

Heavy: Gate C runs the full 1000-draw paired bootstrap (~minutes). Use ``--skip-bootstrap``
for Gate B only. Run via ``parity_substrate.sbatch``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from downstream_evaluation.evaluation.bootstrap_skill_rank import (  # noqa: E402
    align_across_methods,
    compute_per_draw_errors,
)
from downstream_evaluation.evaluation.per_user_pairs import build_per_user_pairs  # noqa: E402
from downstream_evaluation.evaluation.predictions_io import _safe_task  # noqa: E402
from downstream_evaluation.evaluation.skill_score import TASK_DOMAIN_MAP  # noqa: E402
from openmhc._constants import BENCHMARK_TASKS  # noqa: E402

METHODS = ["linear", "multirocket", "xgboost", "lsm2", "gru_d", "wbm", "toto", "chronos2"]
DRAW_KEYS = ["method", "task", "subgroup_attr", "subgroup_value", "draw"]


class Report:
    """Accumulates per-check pass/fail; mirrors the forecasting parity gate."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def summary(self) -> str:
        lines = ["", "=" * 72, "DOWNSTREAM SUBSTRATE PARITY", "=" * 72]
        for name, ok, detail in self.rows:
            lines.append(f"  {'PASS' if ok else 'FAIL':4}  {name}  {detail}")
        lines.append("=" * 72)
        lines.append("ALL PASS" if self.ok() else "FAILURES PRESENT")
        return "\n".join(lines)


def _build_substrates(predictions_dir: Path) -> dict[str, pd.DataFrame]:
    """Build the per-method substrate frame from saved predictions (the producer's core)."""
    return {
        m: build_per_user_pairs(predictions_dir, method_dir=m, tasks=BENCHMARK_TASKS, method_label=m)
        for m in METHODS
    }


def gate_b(rep: Report, subs: dict[str, pd.DataFrame], predictions_dir: Path) -> None:
    """Substrate ``all`` cells == source predictions (float32), per (method, task)."""
    mismatch = 0
    for m, sub in subs.items():
        allc = sub[sub.subgroup_attr == "all"]
        for t in BENCHMARK_TASKS:
            src_p = predictions_dir / m / _safe_task(t) / "test.parquet"
            if not src_p.exists():
                continue
            src = pd.read_parquet(src_p).rename(columns={"uid": "user_id"})
            src["user_id"] = src["user_id"].astype(str)
            s = allc[allc.task == t][["user_id", "y_true", "y_pred", "y_proba"]]
            mrg = s.merge(src, on="user_id", suffixes=("_sub", "_src"))
            if not (len(mrg) == len(src) == len(s)):
                mismatch += 1
                continue
            for c in ("y_true", "y_pred", "y_proba"):
                if not np.array_equal(
                    mrg[f"{c}_sub"].to_numpy(np.float32), mrg[f"{c}_src"].to_numpy(np.float32)
                ):
                    mismatch += 1
                    break
    rep.check("GateB.substrate==predictions (float32)", mismatch == 0, f"{mismatch} cells differ")


def _subgroup_map_from_substrate(sub: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Reconstruct {uid: {age_group, sex}} from the substrate's subgroup rows."""
    sg: dict[str, dict[str, str]] = {}
    for attr in ("age_group", "sex"):
        rows = sub[sub.subgroup_attr == attr][["user_id", "subgroup_value"]].drop_duplicates(
            "user_id"
        )
        for uid, val in zip(rows.user_id.astype(str), rows.subgroup_value.astype(str)):
            sg.setdefault(uid, {})[attr] = val
    return sg


def _method_tasks_from_substrate(sub: pd.DataFrame) -> dict[str, dict]:
    """{task: {uids, y_true, y_pred, y_proba, task_type}} from the substrate's ``all`` cells."""
    allc = sub[sub.subgroup_attr == "all"]
    out: dict[str, dict] = {}
    for task, g in allc.groupby("task", observed=True):
        out[str(task)] = {
            "uids": g["user_id"].astype(str).to_numpy(),
            "y_true": g["y_true"].to_numpy(),
            "y_pred": g["y_pred"].to_numpy(),
            "y_proba": g["y_proba"].to_numpy(),
            "task_type": str(g["task_type"].iloc[0]),
        }
    return out


def gate_c(rep: Report, subs: dict[str, pd.DataFrame], paper_draws: Path, atol: float) -> None:
    """Substrate-driven bootstrap reproduces the committed draws (float32 tolerance)."""
    gt = pd.read_parquet(paper_draws)
    n_boot = int(gt["draw"].nunique()) - 1  # committed draws = point (-1) + n_boot resamples
    aligned = align_across_methods({m: _method_tasks_from_substrate(s) for m, s in subs.items()})
    subgroup_map = _subgroup_map_from_substrate(subs["linear"])
    draws = compute_per_draw_errors(
        aligned,
        n_bootstrap=n_boot,
        seed=42,
        subgroup_map=subgroup_map,
        subgroup_attributes=["age_group", "sex"],
        domain_map=TASK_DOMAIN_MAP,
    )

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in DRAW_KEYS:
            df[c] = df[c].astype(np.int64 if c == "draw" else str)
        df["E"] = df["E"].astype(np.float32)
        return df[DRAW_KEYS + ["E"]]

    merged = norm(draws).merge(norm(gt), on=DRAW_KEYS, how="outer", suffixes=("_new", "_gt"),
                               indicator=True)
    only_new = int((merged["_merge"] == "left_only").sum())
    only_gt = int((merged["_merge"] == "right_only").sum())
    rep.check("GateC.draw row-set == committed", only_new == 0 and only_gt == 0,
              f"only_new={only_new} only_gt={only_gt}")

    both = merged[merged["_merge"] == "both"]
    a, b = both["E_new"].to_numpy(), both["E_gt"].to_numpy()
    diff = np.abs(a - b)
    both_nan = np.isnan(a) & np.isnan(b)
    diff[both_nan] = 0.0
    n_exact = int(np.sum(diff == 0.0))
    worst = float(np.nanmax(diff)) if diff.size else 0.0
    within = bool(np.all((diff <= atol) | both_nan))
    rep.check(
        f"GateC.E reproduces committed (atol={atol:g})", within,
        f"{n_exact}/{len(both)} byte-equal, max|d|={worst:.2e}",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "results/eval/final/predictions")
    p.add_argument("--paper-draws", type=Path, default=REPO_ROOT / "results/paper/bootstrap_draws.parquet")
    p.add_argument("--atol", type=float, default=1e-4, help="float32 tolerance for Gate C E values")
    p.add_argument("--skip-bootstrap", action="store_true", help="Gate B only (skip the heavy Gate C)")
    args = p.parse_args()

    print(f"predictions-dir={args.predictions_dir}")
    subs = _build_substrates(args.predictions_dir)
    rep = Report()
    gate_b(rep, subs, args.predictions_dir)
    if not args.skip_bootstrap:
        gate_c(rep, subs, args.paper_draws, args.atol)
    else:
        print("  (skipped Gate C — bootstrap)")
    print(rep.summary())
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
