#!/usr/bin/env python
r"""Phase 2 of the imputation paper-metrics bootstrap.

Reads ``bootstrap_draws.parquet`` produced by phase 1
(``bootstrap_imputation_draws.py``) and emits the headline sidecar CSVs:

* ``skill_scores_bootstrap.csv``
* ``avg_rankings_bootstrap.csv``

The leaderboard's Fairness Skill Score (disparity-ratio formulation) is
produced by ``aggregate_fairness_skill_score.py`` as a separate sidecar
and is **not** emitted here.

The legacy ``S − λ·D`` fairness-adjusted outputs
(``fairness_subgroup_scores_bootstrap.csv``,
``fairness_summary_bootstrap.csv``) are **deprecated** and no longer
written by default. Pass ``--write-deprecated-fairness`` to opt back in
for back-compat consumers.

Phase 2 is **fast** — every reader passes a different subset of named
disparity functions, λ, or clip bounds without re-resampling.

Example::

    python scripts/paper_results/aggregate_imputation_paper_metrics.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output-dir results/paper/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    aggregate_skill_rank_fairness,
    read_draws_parquet,
)
from imputation_evaluation.evaluation.disparity_metrics import (
    DISPARITY_FUNCTIONS,
    FAIRNESS_COMBINE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: summarise bootstrap_draws.parquet into paper sidecar CSVs",
    )
    p.add_argument(
        "--draws", type=Path, required=True,
        help="Path to bootstrap_draws.parquet from phase 1",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the four sidecar CSVs",
    )
    p.add_argument(
        "--baseline-method", default="locf",
        help="Method to treat as the skill-score baseline (default: locf)",
    )
    p.add_argument(
        "--clip-lower", type=float, default=1e-2,
        help="Lower clip bound for error ratios (default: 1e-2)",
    )
    p.add_argument(
        "--clip-upper", type=float, default=100.0,
        help="Upper clip bound for error ratios (default: 100.0)",
    )
    p.add_argument(
        "--lambda-fairness", type=float, default=0.5,
        help="Lambda for fairness-combine (default: 0.5)",
    )
    p.add_argument(
        "--disparity-fn", action="append", default=None,
        choices=sorted(DISPARITY_FUNCTIONS.keys()),
        help="Named disparity function (repeat to compute several in one pass). "
             "Default: all registered disparities.",
    )
    p.add_argument(
        "--fairness-combine", default="linear_penalty",
        choices=sorted(FAIRNESS_COMBINE.keys()),
        help="Named fairness-combine function (default: linear_penalty)",
    )
    p.add_argument(
        "--ci-level", type=float, default=0.95,
        help="Percentile CI level (default: 0.95)",
    )
    p.add_argument(
        "--method-filter", nargs="+", default=None,
        help="Restrict to these methods only",
    )
    p.add_argument(
        "--write-deprecated-fairness", action="store_true",
        help="Also write the deprecated S − λ·D fairness CSVs "
             "(fairness_subgroup_scores_bootstrap.csv, "
             "fairness_summary_bootstrap.csv). Off by default.",
    )
    return p.parse_args()


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = read_draws_parquet(args.draws)
    logger.info("Loaded %d rows from %s", len(df), args.draws)
    if meta is not None:
        logger.info(
            "Phase-1 meta: n_boot=%s, seed=%s, methods=%s, scenarios=%s",
            meta.get("n_boot"), meta.get("seed"),
            meta.get("methods"), meta.get("scenarios"),
        )
    if args.method_filter:
        df = df[df["method"].isin(args.method_filter)].copy()
        logger.info("After --method-filter: %d rows", len(df))

    if args.disparity_fn:
        disparity_fns = {n: DISPARITY_FUNCTIONS[n].fn for n in args.disparity_fn}
    else:
        disparity_fns = {n: spec.fn for n, spec in DISPARITY_FUNCTIONS.items()}
    logger.info("Disparities: %s", sorted(disparity_fns.keys()))
    logger.info("Fairness-combine: %s, lambda=%s",
                args.fairness_combine, args.lambda_fairness)

    if "rank" not in df.columns:
        logger.error(
            "The Phase-2 reducer requires a 'rank' column on the draws "
            "Parquet, but %s has none. Re-run Phase 1 "
            "(bootstrap_imputation_draws.py) to regenerate draws.",
            args.draws,
        )
        return 2

    tables = aggregate_skill_rank_fairness(
        df,
        baseline_method=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        lambda_fairness=args.lambda_fairness,
        disparity_fns=disparity_fns,
        fairness_combine_name=args.fairness_combine,
        ci_level=args.ci_level,
    )

    out = args.output_dir
    paths = {
        "skill_scores":       out / "skill_scores_bootstrap.csv",
        "avg_rankings":       out / "avg_rankings_bootstrap.csv",
    }
    if args.write_deprecated_fairness:
        paths["fairness_subgroups"] = out / "fairness_subgroup_scores_bootstrap.csv"
        paths["fairness_summary"]   = out / "fairness_summary_bootstrap.csv"
    else:
        logger.info(
            "Skipping deprecated S − λ·D fairness CSVs "
            "(pass --write-deprecated-fairness to opt in). "
            "Leaderboard fairness numbers come from "
            "aggregate_fairness_skill_score.py."
        )
    for key, path in paths.items():
        tbl = tables[key]
        tbl.to_csv(path, index=False, float_format="%.6f")
        logger.info("Wrote %s (%d rows)", path, len(tbl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
