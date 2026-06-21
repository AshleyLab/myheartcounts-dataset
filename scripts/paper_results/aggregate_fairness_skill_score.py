#!/usr/bin/env python
r"""Fairness Skill Score reducer (Phase 2 sidecar).

Reads ``bootstrap_draws.parquet`` produced by ``bootstrap_imputation_draws.py``
and emits ``fairness_skill_score_bootstrap.csv``: per-method fairness skill
score for each sensitive attribute (sex, age_group) plus the macro-averaged
overall score, all with mean / SE / percentile CI across draws.

Formulation (mirrors the regular skill score machinery):

    For each task r = (scenario, channel) and attribute G ∈ {sex, age_group}:
        D_{r,j}^{(G)}  =  (2 / |G|(|G|-1)) · Σ_{g, g' ∈ G, g ≠ g'}
                                  | E_{r,j}^{(g)}  −  E_{r,j}^{(g')} |
        D_{r,b}^{(G)}  =  same, for the baseline method b (= LOCF).
        drop r from this (G) aggregation if  D_{r,b}^{(G)} ≤ 0  or NaN.
        ratio_r        =  clip( D_{r,j}^{(G)} / D_{r,b}^{(G)},  ℓ,  u )

    The disparity is the mean absolute pairwise difference (MAPD) over
    the common subgroup set — averaged across the n(n-1)/2 unordered
    (g, g') pairs. For |G| = 2 (e.g. sex) this collapses to
    ``|E_a − E_b|``, matching the historical max-min formulation; for
    |G| ≥ 3 (e.g. age_group with 5 buckets) MAPD smooths over every
    pair instead of only the two extremes.

    Per attribute:
        S^{(G)}_j      =  1  −  GeometricMean_r(ratio_r)
                       =  1  −  exp( mean_r log(ratio_r) )

    Macro-average across attributes:
        S_fair_j       =  (1 / |A|) · Σ_{G ∈ A} S^{(G)}_j

The ``unknown`` bucket is preserved as a structural subgroup per the
appendix; it contributes to the per-task pairwise differences like any
other subgroup.

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

from imputation_evaluation.evaluation.bca import (
    _augment_with_bca,
    _draws_by_key,
    _pad_jackknife_maps,
)
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    read_draws_parquet,
    read_per_user_errors_parquet,
)
from imputation_evaluation.evaluation.paper_metrics_core import (
    _per_attribute_skill_keyed,
    compute_fair_skill_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


SENSITIVE_ATTRS = ("age_group", "sex")
OVERALL_SCOPE = "overall"

# Headline scopes that receive a point estimate + BCa interval (the rest keep the
# percentile CI only). The 3 published fairness scopes are all headline; the
# disparity ratio max_g E - min_g E is skewed and downward-biased so the BCa
# re-anchoring matters. See METRICS.md §S7.
BCA_HEADLINE_SCOPES: frozenset[str] = frozenset({OVERALL_SCOPE, *SENSITIVE_ATTRS})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute fairness skill scores (per-attribute and macro-averaged) "
            "from bootstrap_draws.parquet."
        ),
    )
    p.add_argument(
        "--draws",
        type=Path,
        required=True,
        help="Path to bootstrap_draws.parquet from Phase 1.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (typically fairness_skill_score_bootstrap.csv).",
    )
    p.add_argument(
        "--baseline-method",
        default="locf",
        help="Baseline model for the disparity ratio denominator (default: locf).",
    )
    p.add_argument(
        "--clip-lower",
        type=float,
        default=1e-2,
        help="Lower clip bound for disparity ratios (default: 1e-2).",
    )
    p.add_argument(
        "--clip-upper",
        type=float,
        default=100.0,
        help="Upper clip bound for disparity ratios (default: 100.0).",
    )
    p.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Percentile CI level (default: 0.95).",
    )
    p.add_argument(
        "--attrs",
        nargs="+",
        default=list(SENSITIVE_ATTRS),
        help=("Sensitive attributes to include in the macro-average (default: age_group sex)."),
    )
    p.add_argument(
        "--method-filter",
        nargs="+",
        default=None,
        help=(
            "Restrict to these methods only. Fairness skill values are "
            "pairwise vs. the baseline so values stay the same as the "
            "full-pool run; the baseline method (default: locf) MUST be "
            "in the filter or fairness rows will be empty. See METRICS.md "
            "§8.1 for the full subset-recompute workflow."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (non-zero exit) on any sensitive attribute that is missing, "
            "degenerate, or yields no usable tasks instead of warning-and-"
            "skipping. Required for runs whose numbers are published."
        ),
    )
    bca_grp = p.add_mutually_exclusive_group()
    bca_grp.add_argument(
        "--bca",
        dest="bca",
        action="store_true",
        default=True,
        help=(
            "Emit point + BCa (bias-corrected & accelerated) CI columns for "
            "the headline fairness scopes (default ON). The disparity ratio "
            "max_g E - min_g E is downward-biased; BCa re-anchors the interval "
            "at the deterministic point. See METRICS.md §S7."
        ),
    )
    bca_grp.add_argument(
        "--no-bca",
        dest="bca",
        action="store_false",
        help="Disable BCa augmentation; emit only the legacy percentile columns.",
    )
    p.add_argument(
        "--per-user-errors",
        type=Path,
        default=None,
        help=(
            "Path to per_user_errors.parquet from Phase 1, required when --bca "
            "is on. Defaults to '<draws>.parent / per_user_errors.parquet'."
        ),
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

    Thin wrapper around ``paper_metrics_core._per_attribute_skill_keyed``
    that adds ``draw`` as an extra grouping key, so the bootstrap CSV and
    the deterministic ``fairness_skill_scores.csv`` from
    ``compute_imputation_paper_metrics.py`` share a single source of truth.

    Returns a frame with columns ``method, draw, S_attr, n_tasks``.
    """
    return _per_attribute_skill_keyed(
        df_attr,
        extra_keys=["draw"],
        baseline_method=baseline_method,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


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


_PER_CELL_GROUP_COLS = [
    "method",
    "scenario",
    "channel",
    "channel_type",
    "subgroup_attr",
    "subgroup_value",
]


def _per_user_to_per_cell_E(per_user_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-user E to per-cell E (user-macro mean per cell).

    Output schema matches what ``compute_fair_skill_scores`` consumes:
    ``[method, scenario, channel, channel_type, subgroup_attr,
    subgroup_value, E]``. The mean equals the identity bootstrap draw, so
    BCa's point estimate matches the deterministic point flow.
    """
    if per_user_df.empty:
        return pd.DataFrame(columns=[*_PER_CELL_GROUP_COLS, "E"])
    grouped = (
        per_user_df.groupby(_PER_CELL_GROUP_COLS, observed=True)["E_per_user"]
        .mean()
        .reset_index()
        .rename(columns={"E_per_user": "E"})
    )
    return grouped


def _fair_points_by_key(
    per_cell_df: pd.DataFrame,
    *,
    attrs: list[str],
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
    scopes: frozenset[str],
) -> dict[tuple, float]:
    """Run ``compute_fair_skill_scores`` and key it by ``(method, scope)``."""
    fair = compute_fair_skill_scores(
        per_cell_df,
        attrs=attrs,
        baseline_method=baseline_method,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    out: dict[tuple, float] = {}
    for _, row in fair.iterrows():
        scope = str(row["scope"])
        if scope not in scopes:
            continue
        val = row["fair_skill_score"]
        if pd.notna(val):
            out[(str(row["method"]), scope)] = float(val)
    return out


def _jackknife_fair_points_from_per_user(
    per_user_df: pd.DataFrame,
    *,
    attrs: list[str],
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
    scopes: frozenset[str],
) -> dict[tuple, np.ndarray]:
    """Leave-one-user-out jackknife of the fair-skill headline scopes.

    Drops one user from ``per_user_df`` at a time, re-collapses to per-cell E,
    re-runs ``compute_fair_skill_scores``, and gathers ``(method, scope) ->
    fair_skill_score`` per user. Returns ``{(method, scope): array of length
    n_users}`` with NaN where a scope is absent for that user's recompute.

    The B.2 two-stage category-balanced aggregation lives inside
    ``_per_attribute_skill_keyed``; the jackknife wraps it, so the bucket
    logic runs unchanged inside each LOO recompute.
    """
    if per_user_df.empty:
        return {}
    uid_arr = per_user_df["user_id"].astype(str).to_numpy()
    users = sorted(set(uid_arr))
    per_user_maps: list[dict[tuple, float]] = []
    for user in users:
        loo = per_user_df.loc[uid_arr != user]
        per_cell = _per_user_to_per_cell_E(loo)
        per_user_maps.append(
            _fair_points_by_key(
                per_cell,
                attrs=attrs,
                baseline_method=baseline_method,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                scopes=scopes,
            )
        )
    return _pad_jackknife_maps(per_user_maps)


def compute_fairness_skill_scores(
    draws_df: pd.DataFrame,
    *,
    attrs: list[str],
    baseline_method: str = "locf",
    clip_lower: float = 1e-2,
    clip_upper: float = 100.0,
    ci_level: float = 0.95,
    strict: bool = False,
    bca: bool = False,
    per_user_df: pd.DataFrame | None = None,
    bca_scopes: frozenset[str] = BCA_HEADLINE_SCOPES,
) -> pd.DataFrame:
    """End-to-end: per-attribute + macro-averaged fairness skill score.

    Returns one row per (method, scope, split) with columns
    ``method, scope, split, n_tasks, mean, se, ci_lo, ci_hi, n_boot``.
    Scope ∈ {one per attribute, plus ``"overall"`` for the macro-average}.

    When ``bca=True``, three additional columns ``point, bca_lo, bca_hi``
    are appended; ``point`` is filled for every row, ``bca_lo``/``bca_hi``
    are filled only for rows whose ``scope`` is in ``bca_scopes`` (NaN
    elsewhere). Requires ``per_user_df`` (the Phase 1 sibling Parquet).
    """
    if bca and per_user_df is None:
        raise ValueError("compute_fairness_skill_scores(bca=True) requires per_user_df")
    splits = sorted(draws_df["split"].unique())
    if len(splits) > 1:
        logger.warning(
            "draws_df has multiple splits %s — aggregating each independently",
            splits,
        )

    summary_frames: list[pd.DataFrame] = []
    # When bca=True, collect per-draw S values keyed by (method, scope, split)
    # — fed to _augment_with_bca after the percentile summary is built.
    draws_records: list[dict] = []
    point_by_key: dict[tuple, float] = {}
    jack_by_key: dict[tuple, np.ndarray] = {}

    # Fairness B.2: keep continuous per-channel rows (activity / physiology
    # buckets) AND the cat_collapsed:{sleep,workouts} rows. Drop per-channel
    # binary ch_7..ch_18 rows — the sleep / workouts buckets reach the
    # fairness headline only via the collapsed rows, so per-channel binary
    # would double-count. ``_per_attribute_skill_keyed`` enforces the same
    # rule downstream via ``b2_bucket_for_channel`` (rows it would drop are
    # filtered to ``bucket = None``), but stripping them up front cuts the
    # join cost and surfaces the row-count drop in the log.
    is_per_channel_binary = draws_df["channel"].astype(str).str.match(r"^ch_(?:[7-9]|1[0-8])$") & (
        draws_df["channel_type"].astype(str) == "binary"
    )
    if is_per_channel_binary.any():
        logger.info(
            "Fairness B.2: dropping %d per-channel binary rows "
            "(replaced by cat_collapsed:{sleep,workouts})",
            int(is_per_channel_binary.sum()),
        )
    draws_df = draws_df[~is_per_channel_binary]

    if bca and per_user_df is not None:
        # Same B.2 row-stripping on the per-user side so the point + jackknife
        # match the percentile-CI denominator exactly. (The point flow's own
        # ``b2_bucket_for_channel`` would also filter these, but stripping up
        # front mirrors the draws_df path and cuts the recompute cost.)
        is_per_channel_binary_pu = per_user_df["channel"].astype(str).str.match(
            r"^ch_(?:[7-9]|1[0-8])$"
        ) & (per_user_df["channel_type"].astype(str) == "binary")
        if is_per_channel_binary_pu.any():
            per_user_df = per_user_df[~is_per_channel_binary_pu]

    for split in splits:
        df_split = draws_df[draws_df["split"] == split]

        per_attr_results: dict[str, pd.DataFrame] = {}
        for attr in attrs:
            df_attr = df_split[df_split["subgroup_attr"] == attr]
            if df_attr.empty:
                msg = f"[split={split}] no rows for attribute {attr!r}"
                if strict:
                    raise RuntimeError(f"[strict] {msg} — aborting")
                logger.warning("%s — skipping", msg)
                continue
            n_subgroups = df_attr["subgroup_value"].nunique()
            if n_subgroups < 2:
                msg = (
                    f"[split={split}] attribute {attr!r} has only "
                    f"{n_subgroups} subgroup value(s) — max-min disparity is "
                    f"degenerate"
                )
                if strict:
                    raise RuntimeError(f"[strict] {msg} — aborting")
                logger.warning("%s; skipping.", msg)
                continue

            per_draw = _per_attribute_skill(
                df_attr,
                baseline_method=baseline_method,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
            )
            if per_draw.empty:
                msg = (
                    f"[split={split}] attribute {attr!r} yielded no usable "
                    f"tasks after dropping D_b<=0"
                )
                if strict:
                    raise RuntimeError(f"[strict] {msg} — aborting")
                logger.warning("%s; skipping.", msg)
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

            if bca:
                for _, row in per_draw.iterrows():
                    draws_records.append(
                        {
                            "method": str(row["method"]),
                            "scope": attr,
                            "split": split,
                            "value": float(row["S_attr"]),
                        }
                    )

        # Macro-average across attributes (arithmetic mean of per-attribute
        # S^{(G)} per (method, draw)). Methods or draws missing from any
        # attribute drop out of the overall row to keep the average honest.
        if per_attr_results:
            stacked = pd.concat(
                [df.assign(attr=attr_name) for attr_name, df in per_attr_results.items()],
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

            if bca:
                for _, row in overall.iterrows():
                    draws_records.append(
                        {
                            "method": str(row["method"]),
                            "scope": OVERALL_SCOPE,
                            "split": split,
                            "value": float(row["S_fair"]),
                        }
                    )

        if bca and per_user_df is not None:
            # Point + LOO jackknife on this split only.
            pu_split = per_user_df[per_user_df["split"] == split]
            if not pu_split.empty:
                per_cell_split = _per_user_to_per_cell_E(pu_split)
                pts = _fair_points_by_key(
                    per_cell_split,
                    attrs=attrs,
                    baseline_method=baseline_method,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                    scopes=bca_scopes,
                )
                jack = _jackknife_fair_points_from_per_user(
                    pu_split,
                    attrs=attrs,
                    baseline_method=baseline_method,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                    scopes=bca_scopes,
                )
                for (method, scope), value in pts.items():
                    point_by_key[(method, scope, split)] = value
                for (method, scope), arr in jack.items():
                    jack_by_key[(method, scope, split)] = arr

    base_cols = [
        "method",
        "scope",
        "split",
        "n_tasks",
        "mean",
        "se",
        "ci_lo",
        "ci_hi",
        "n_boot",
    ]
    if not summary_frames:
        empty = pd.DataFrame(columns=base_cols)
        if bca:
            for col in ("point", "bca_lo", "bca_hi"):
                empty[col] = pd.Series(dtype=np.float64)
        return empty
    out = pd.concat(summary_frames, ignore_index=True)[base_cols]

    if bca:
        out = _augment_with_bca(
            out,
            draws_by_key=_draws_by_key(draws_records, ["method", "scope", "split"]),
            point_by_key=point_by_key,
            jack_by_key=jack_by_key,
            scopes=bca_scopes,
            ci_level=ci_level,
            key_cols=["method", "scope", "split"],
        )
    return out


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df, meta = read_draws_parquet(args.draws)
    logger.info("Loaded %d rows from %s", len(df), args.draws)
    if meta is not None:
        logger.info(
            "Phase-1 meta: n_boot=%s, seed=%s, methods=%d, scenarios=%s",
            meta.get("n_boot"),
            meta.get("seed"),
            len(meta.get("methods", [])),
            meta.get("scenarios"),
        )
    if args.method_filter:
        df = df[df["method"].isin(args.method_filter)].copy()
        logger.info("After --method-filter: %d rows", len(df))

    per_user_df: pd.DataFrame | None = None
    if args.bca:
        per_user_path = (
            args.per_user_errors
            if args.per_user_errors is not None
            else args.draws.parent / "per_user_errors.parquet"
        )
        if not per_user_path.exists():
            logger.error(
                "--bca requires %s but the file does not exist; pass "
                "--per-user-errors PATH or rerun Phase 1 without "
                "--no-per-user-errors, or pass --no-bca.",
                per_user_path,
            )
            return 2
        per_user_df, _pu_meta = read_per_user_errors_parquet(per_user_path)
        logger.info(
            "Loaded %d per-user rows from %s",
            len(per_user_df),
            per_user_path,
        )
        if args.method_filter:
            per_user_df = per_user_df[per_user_df["method"].isin(args.method_filter)].copy()
            logger.info("After --method-filter (per-user): %d rows", len(per_user_df))

    out_df = compute_fairness_skill_scores(
        df,
        attrs=args.attrs,
        baseline_method=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
        strict=args.strict,
        bca=args.bca,
        per_user_df=per_user_df,
    )
    out_df.to_csv(args.output, index=False, float_format="%.6f")
    logger.info("Wrote %s (%d rows)", args.output, len(out_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
