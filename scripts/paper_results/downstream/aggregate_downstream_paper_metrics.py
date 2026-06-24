#!/usr/bin/env python
r"""Phase 2 of the downstream paper-metrics bootstrap.

Reads ``bootstrap_draws.parquet`` produced by phase 1
(``bootstrap_downstream_draws.py``) and emits the four sidecar CSVs that carry
mean / SE / 95 % CI for the headline metrics:

* ``skill_scores_bootstrap.csv``            — per method × scope (Overall + per-domain)
* ``avg_rankings_bootstrap.csv``            — per method × scope
* ``fairness_subgroup_scores_bootstrap.csv``— per method × demographic_attr × subgroup
* ``task_metrics_bootstrap.csv``            — per method × task: the headline metric
  (auprc / spearman_r / pearson_r) with full-cohort point + SE + percentile CI

The headline fairness metric (the disparity-ratio Fairness Skill Score) is
produced separately by ``aggregate_fairness_skill_score.py``.

Phase 2 is **fast** — re-run with different clip bounds or CI level without
re-resampling.

Usage::

    PYTHONPATH=src python scripts/paper_results/downstream/aggregate_downstream_paper_metrics.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output-dir results/paper/ \
        --baseline-method linear
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    POINT_DRAW,
    _summarise,
    aggregate_skill_rank_fairness,
    read_draws_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Task type → its headline metric name. The benchmark is binary / ordinal / regression
# only (no multiclass), so those three are handled; an unexpected task_type is a hard
# error rather than a silently mislabeled row.
PRIMARY_METRIC = {
    "binary": "auprc",
    "ordinal": "spearman_r",
    "regression": "pearson_r",
}


def summarise_task_metrics(draws: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    """Per-(method, task) global headline metric: full-cohort point + bootstrap SE + CI.

    Reuses the phase-1 per-draw errors (``E = 1 - metric`` at global scope,
    ``subgroup_attr == "all"``) and the shared ``_summarise`` helper, so the per-task CI
    uses the same convention as the skill / rank / fairness tables. The ``POINT_DRAW`` row
    is the full-cohort estimate reported as ``value``; the ``draw >= 0`` rows give the
    SE / CI around it. Returns one row per (method, task).
    """
    glob = draws[draws["subgroup_attr"] == "all"]
    rows: list[dict] = []
    for (method, task), g in glob.groupby(["method", "task"], sort=True):
        task_type = str(g["task_type"].iloc[0])
        if task_type not in PRIMARY_METRIC:
            raise ValueError(
                f"task {task!r} has unhandled task_type {task_type!r}; "
                f"expected one of {sorted(PRIMARY_METRIC)}"
            )
        point_rows = g[g["draw"] == POINT_DRAW]
        boot = g[g["draw"] != POINT_DRAW]
        # metric = 1 - E; the point draw is the full-cohort estimate reported as `value`.
        point = (1.0 - float(point_rows["E"].iloc[0])) if len(point_rows) else None
        metric_draws = (1.0 - boot["E"].to_numpy(dtype=float)).tolist()
        stats = _summarise(metric_draws, ci_level, point=point)
        rows.append(
            {
                "method": method,
                "task": task,
                "task_type": task_type,
                "metric": PRIMARY_METRIC[task_type],
                "value": stats["point"],
                "se": stats["se"],
                "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=["method", "task", "task_type", "metric", "value", "se", "ci_lo", "ci_hi"],
    )


def main() -> int:
    """Summarise ``bootstrap_draws.parquet`` into the four paper sidecar CSVs."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--draws", type=Path, required=True, help="bootstrap_draws.parquet from phase 1")
    p.add_argument("--output-dir", type=Path, required=True, help="Dir for the four sidecar CSVs")
    p.add_argument(
        "--baseline-method",
        default="linear",
        help="Method treated as the skill-score baseline (default: linear)",
    )
    p.add_argument("--clip-lower", type=float, default=1e-2)
    p.add_argument("--clip-upper", type=float, default=100.0)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--method-filter", nargs="+", default=None, help="Restrict to these methods.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = read_draws_parquet(args.draws)
    log.info("Loaded %d rows from %s", len(df), args.draws)
    if meta is not None:
        log.info(
            "Phase-1 meta: n_boot=%s, seed=%s, methods=%s",
            meta.get("n_boot"),
            meta.get("seed"),
            meta.get("methods"),
        )
    if args.method_filter:
        df = df[df["method"].isin(args.method_filter)].copy()
        log.info("After --method-filter: %d rows", len(df))

    tables = aggregate_skill_rank_fairness(
        df,
        baseline=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
    )
    tables["task_metrics"] = summarise_task_metrics(df, ci_level=args.ci_level)

    paths = {
        "skill_scores": args.output_dir / "skill_scores_bootstrap.csv",
        "avg_rankings": args.output_dir / "avg_rankings_bootstrap.csv",
        "fairness_subgroup_scores": args.output_dir / "fairness_subgroup_scores_bootstrap.csv",
        "task_metrics": args.output_dir / "task_metrics_bootstrap.csv",
    }
    for key, path in paths.items():
        tbl = tables[key]
        tbl.to_csv(path, index=False, float_format="%.6f")
        log.info("Wrote %s (%d rows)", path, len(tbl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
