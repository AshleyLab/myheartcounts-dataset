#!/usr/bin/env python
r"""Fairness Skill Score reducer (Phase 2 sidecar).

Reads ``bootstrap_draws.parquet`` produced by ``bootstrap_imputation_draws.py``
and emits ``fairness_skill_score_bootstrap.csv``: per-method fairness skill
score for each sensitive attribute (sex, age_group) plus the macro-averaged
overall score, all with mean / SE / percentile CI across draws.

Formulation (mirrors the regular skill score machinery):

    For each task r = (scenario, channel) and attribute G ∈ {sex, age_group}:
        D_{r,j}^{(G)}  =  max_g E_{r,j}^{(g)}  −  min_g E_{r,j}^{(g)}
        D_{r,b}^{(G)}  =  max_g E_{r,b}^{(g)}  −  min_g E_{r,b}^{(g)}    (b = LOCF)
        drop r from this (G) aggregation if  D_{r,b}^{(G)} ≤ 0  or NaN.
        ratio_r        =  clip( D_{r,j}^{(G)} / D_{r,b}^{(G)},  ℓ,  u )

    Per attribute:
        S^{(G)}_j      =  1  −  GeometricMean_r(ratio_r)
                       =  1  −  exp( mean_r log(ratio_r) )

    Macro-average across attributes:
        S_fair_j       =  (1 / |A|) · Σ_{G ∈ A} S^{(G)}_j

The ``unknown`` bucket is preserved as a structural subgroup per the
appendix; it contributes to the per-task max-min like any other subgroup.

Bootstrapping: per-draw D_j and D_b share the same resampled cohort (the
``draw`` axis), so pairing is preserved. Summary statistics aggregate
across draws: mean, standard error, percentile CI.

Example::

    python scripts/paper_results/aggregate_fairness_skill_score.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output results/paper/fairness_skill_score_bootstrap.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.bootstrap_skill_rank import read_draws_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


SENSITIVE_ATTRS = ("age_group", "sex")
OVERALL_SCOPE = "overall"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute fairness skill scores (per-attribute and macro-averaged) "
            "from bootstrap_draws.parquet."
        ),
    )
    p.add_argument(
        "--draws", type=Path, required=True,
        help="Path to bootstrap_draws.parquet from Phase 1.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path (typically fairness_skill_score_bootstrap.csv).",
    )
    p.add_argument(
        "--baseline-method", default="locf",
        help="Baseline model for the disparity ratio denominator (default: locf).",
    )
    p.add_argument(
        "--clip-lower", type=float, default=1e-2,
        help="Lower clip bound for disparity ratios (default: 1e-2).",
    )
    p.add_argument(
        "--clip-upper", type=float, default=100.0,
        help="Upper clip bound for disparity ratios (default: 100.0).",
    )
    p.add_argument(
        "--ci-level", type=float, default=0.95,
        help="Percentile CI level (default: 0.95).",
    )
    p.add_argument(
        "--attrs", nargs="+", default=list(SENSITIVE_ATTRS),
        help=(
            "Sensitive attributes to include in the macro-average "
            "(default: age_group sex)."
        ),
    )
    p.add_argument(
        "--method-filter", nargs="+", default=None,
        help="Restrict to these methods only.",
    )
    return p.parse_args()


def _per_attribute_skill(
    df_attr: pd.DataFrame,
    *,
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Per-(method, draw) fairness skill score for a single attribute.

    ``df_attr`` is the subset of the draws table for one attribute (one
    ``subgroup_attr`` value other than ``all``).

    Returns a frame with columns ``method, draw, S_attr, n_tasks``.
    """
    # Per (draw, method, scenario, channel): D = max_g E - min_g E across
    # subgroup values. Vectorised across all draws at once.
    grouped = df_attr.groupby(
        ["draw", "method", "scenario", "channel", "channel_type"],
        observed=True,
    )["E"]
    D_max = grouped.max()
    D_min = grouped.min()
    D = (D_max - D_min).rename("D").reset_index()

    # Split baseline rows from model rows on the same (draw, scenario,
    # channel) keys; merge so each model row carries its paired D_b.
    # Keep the baseline on both sides so its self-ratio (D_b/D_b = 1, clipped
    # → S = 0) lands in the output for parity with compute_skill_scores'
    # treatment of LOCF in skill_scores_bootstrap.csv.
    bl = (
        D[D["method"] == baseline_method]
        .drop(columns=["method"])
        .rename(columns={"D": "D_b"})
    )
    jm = D.rename(columns={"D": "D_j"})
    merged = jm.merge(
        bl,
        on=["draw", "scenario", "channel", "channel_type"],
        how="inner",
    )

    # Drop tasks where the baseline is already perfectly fair (D_b ≤ 0)
    # or where either disparity is NaN. Mirrors compute_skill_scores'
    # ``e_baseline <= 0 or np.isnan(e_baseline)`` drop rule.
    keep = (
        (merged["D_b"] > 0)
        & merged["D_b"].notna()
        & merged["D_j"].notna()
        & (merged["D_j"] >= 0)  # max-min is non-negative by construction
    )
    merged = merged.loc[keep].copy()
    if merged.empty:
        return pd.DataFrame(columns=["method", "draw", "S_attr", "n_tasks"])

    # Clip the ratio then take the per-(method, draw) geometric mean via
    # log-mean-exp. The skill score is 1 - GM(clipped_ratios).
    ratio = (merged["D_j"] / merged["D_b"]).clip(
        lower=clip_lower, upper=clip_upper,
    )
    merged["log_ratio"] = np.log(ratio.to_numpy())

    agg = (
        merged.groupby(["method", "draw"], observed=True)
        .agg(log_ratio_mean=("log_ratio", "mean"), n_tasks=("log_ratio", "size"))
        .reset_index()
    )
    agg["S_attr"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[["method", "draw", "S_attr", "n_tasks"]]


def _summarise_across_draws(
    per_draw: pd.DataFrame,
    *,
    value_col: str,
    ci_level: float,
    key_cols: list[str],
    n_tasks_col: str | None,
) -> pd.DataFrame:
    """Reduce a per-(key, draw) frame to mean / SE / percentile CI per key."""
    alpha = 1.0 - ci_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    rows = []
    for keys, grp in per_draw.groupby(key_cols, observed=True):
        values = grp[value_col].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean = se = ci_lo = ci_hi = float("nan")
        else:
            mean = float(np.mean(finite))
            se = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            ci_lo = float(np.percentile(finite, lo_q))
            ci_hi = float(np.percentile(finite, hi_q))
        row = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        # n_tasks is a per-draw count that should be (approximately) constant
        # per group; report the median across draws for robustness.
        if n_tasks_col and n_tasks_col in grp.columns:
            row["n_tasks"] = int(np.median(grp[n_tasks_col].to_numpy()))
        row["mean"] = mean
        row["se"] = se
        row["ci_lo"] = ci_lo
        row["ci_hi"] = ci_hi
        row["n_boot"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_fairness_skill_scores(
    draws_df: pd.DataFrame,
    *,
    attrs: list[str],
    baseline_method: str = "locf",
    clip_lower: float = 1e-2,
    clip_upper: float = 100.0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """End-to-end: per-attribute + macro-averaged fairness skill score.

    Returns one row per (method, scope, split) with columns
    ``method, scope, split, n_tasks, mean, se, ci_lo, ci_hi, n_boot``.
    Scope ∈ {one per attribute, plus ``"overall"`` for the macro-average}.
    """
    splits = sorted(draws_df["split"].unique())
    if len(splits) > 1:
        logger.warning(
            "draws_df has multiple splits %s — aggregating each independently",
            splits,
        )

    summary_frames: list[pd.DataFrame] = []

    for split in splits:
        df_split = draws_df[draws_df["split"] == split]

        per_attr_results: dict[str, pd.DataFrame] = {}
        for attr in attrs:
            df_attr = df_split[df_split["subgroup_attr"] == attr]
            if df_attr.empty:
                logger.warning(
                    "[split=%s] no rows for attribute %r — skipping",
                    split, attr,
                )
                continue
            n_subgroups = df_attr["subgroup_value"].nunique()
            if n_subgroups < 2:
                logger.warning(
                    "[split=%s] attribute %r has only %d subgroup value(s) — "
                    "max-min disparity is degenerate; skipping.",
                    split, attr, n_subgroups,
                )
                continue

            per_draw = _per_attribute_skill(
                df_attr,
                baseline_method=baseline_method,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
            )
            if per_draw.empty:
                logger.warning(
                    "[split=%s] attribute %r yielded no usable tasks after "
                    "dropping D_b<=0; skipping.",
                    split, attr,
                )
                continue
            per_attr_results[attr] = per_draw

            attr_summary = _summarise_across_draws(
                per_draw,
                value_col="S_attr",
                ci_level=ci_level,
                key_cols=["method"],
                n_tasks_col="n_tasks",
            )
            attr_summary["scope"] = attr
            attr_summary["split"] = split
            summary_frames.append(attr_summary)

        # Macro-average across attributes (arithmetic mean of per-attribute
        # S^{(G)} per (method, draw)). Methods or draws missing from any
        # attribute drop out of the overall row to keep the average honest.
        if per_attr_results:
            stacked = pd.concat(
                [
                    df.assign(attr=attr_name)
                    for attr_name, df in per_attr_results.items()
                ],
                ignore_index=True,
            )
            n_attrs_seen = (
                stacked.groupby(["method", "draw"], observed=True)["attr"]
                .nunique()
                .reset_index(name="n_attrs")
            )
            full_coverage = n_attrs_seen[n_attrs_seen["n_attrs"] == len(per_attr_results)]
            stacked = stacked.merge(
                full_coverage[["method", "draw"]],
                on=["method", "draw"],
                how="inner",
            )
            overall = (
                stacked.groupby(["method", "draw"], observed=True)
                .agg(
                    S_fair=("S_attr", "mean"),
                    n_tasks=("n_tasks", "sum"),
                )
                .reset_index()
            )
            overall_summary = _summarise_across_draws(
                overall,
                value_col="S_fair",
                ci_level=ci_level,
                key_cols=["method"],
                n_tasks_col="n_tasks",
            )
            overall_summary["scope"] = OVERALL_SCOPE
            overall_summary["split"] = split
            summary_frames.append(overall_summary)

    if not summary_frames:
        return pd.DataFrame(
            columns=[
                "method", "scope", "split", "n_tasks",
                "mean", "se", "ci_lo", "ci_hi", "n_boot",
            ]
        )
    out = pd.concat(summary_frames, ignore_index=True)
    return out[
        [
            "method", "scope", "split", "n_tasks",
            "mean", "se", "ci_lo", "ci_hi", "n_boot",
        ]
    ]


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df, meta = read_draws_parquet(args.draws)
    logger.info("Loaded %d rows from %s", len(df), args.draws)
    if meta is not None:
        logger.info(
            "Phase-1 meta: n_boot=%s, seed=%s, methods=%d, scenarios=%s",
            meta.get("n_boot"), meta.get("seed"),
            len(meta.get("methods", [])), meta.get("scenarios"),
        )
    if args.method_filter:
        df = df[df["method"].isin(args.method_filter)].copy()
        logger.info("After --method-filter: %d rows", len(df))

    out_df = compute_fairness_skill_scores(
        df,
        attrs=args.attrs,
        baseline_method=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
    )
    out_df.to_csv(args.output, index=False, float_format="%.6f")
    logger.info("Wrote %s (%d rows)", args.output, len(out_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
