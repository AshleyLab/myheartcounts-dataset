#!/usr/bin/env python
r"""Phase 2 of the downstream paper-metrics bootstrap.

Reads ``bootstrap_draws.parquet`` produced by phase 1
(``bootstrap_downstream_draws.py``) and emits the four sidecar CSVs that carry
mean / SE / 95 % CI for the headline metrics:

* ``skill_scores_bootstrap.csv``            — per method × scope (Overall + per-domain)
* ``avg_rankings_bootstrap.csv``            — per method × scope
* ``fairness_subgroup_scores_bootstrap.csv``— per method × demographic_attr × subgroup
* ``fairness_summary_bootstrap.csv``        — per method: S_overall, disparity, fairness-adjusted

Phase 2 is **fast** — re-run with a different subset of disparity functions, λ,
or clip bounds without re-resampling.

Usage::

    PYTHONPATH=src python scripts/paper_results/aggregate_downstream_paper_metrics.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output-dir results/paper/ \
        --baseline-method stat_simple \
        --disparity-fn max_minus_min --disparity-fn worst_group \
        --lambda-fairness 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    aggregate_skill_rank_fairness,
    read_draws_parquet,
)
from downstream_evaluation.evaluation.disparity_metrics import (
    DISPARITY_FUNCTIONS,
    FAIRNESS_COMBINE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


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
        default="stat_simple",
        help="Method treated as the skill-score baseline (default: stat_simple)",
    )
    p.add_argument("--clip-lower", type=float, default=1e-2)
    p.add_argument("--clip-upper", type=float, default=100.0)
    p.add_argument("--lambda-fairness", type=float, default=0.5)
    p.add_argument(
        "--disparity-fn",
        action="append",
        default=None,
        choices=sorted(DISPARITY_FUNCTIONS.keys()),
        help="Named disparity function (repeatable). Default: all registered.",
    )
    p.add_argument(
        "--fairness-combine", default="linear_penalty", choices=sorted(FAIRNESS_COMBINE.keys())
    )
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

    if args.disparity_fn:
        disparity_fns = {n: DISPARITY_FUNCTIONS[n].fn for n in args.disparity_fn}
    else:
        disparity_fns = {n: spec.fn for n, spec in DISPARITY_FUNCTIONS.items()}
    log.info(
        "Disparities: %s | combine=%s lambda=%s",
        sorted(disparity_fns),
        args.fairness_combine,
        args.lambda_fairness,
    )

    tables = aggregate_skill_rank_fairness(
        df,
        baseline=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        lambda_fairness=args.lambda_fairness,
        disparity_fns=disparity_fns,
        fairness_combine_name=args.fairness_combine,
        ci_level=args.ci_level,
    )

    paths = {
        "skill_scores": args.output_dir / "skill_scores_bootstrap.csv",
        "avg_rankings": args.output_dir / "avg_rankings_bootstrap.csv",
        "fairness_subgroup_scores": args.output_dir / "fairness_subgroup_scores_bootstrap.csv",
        "fairness_summary": args.output_dir / "fairness_summary_bootstrap.csv",
    }
    for key, path in paths.items():
        tbl = tables[key]
        tbl.to_csv(path, index=False, float_format="%.6f")
        log.info("Wrote %s (%d rows)", path, len(tbl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
