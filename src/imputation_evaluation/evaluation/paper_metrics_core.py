"""Pure functions for the imputation paper's headline metrics.

Holds the constants (clip bounds, baselines, semantic-scenario set,
channel categories) and the deterministic transforms used to derive
**skill score**, **average rank**, and **baseline errors** from a long-format
``errors`` DataFrame.

Both the canonical script
``scripts/paper_results/compute_imputation_paper_metrics.py`` and the
bootstrap pipeline (``bootstrap_skill_rank.py``) import from this module so
the per-draw aggregation matches the published point estimates exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

CLIP_LOWER = 1e-2
CLIP_UPPER = 100.0

BASELINE_CONTINUOUS = "locf"
BASELINE_BINARY = "locf"

DEFAULT_LAMBDA = 0.5

# Disparity-ratio fair skill score (the metric quoted in the leaderboard's
# ``fair_skill`` column). See ``compute_fair_skill_scores`` for the full math.
DEFAULT_FAIRNESS_ATTRS: tuple[str, ...] = ("age_group", "sex")
FAIRNESS_OVERALL_SCOPE = "overall"

N_CONTINUOUS = 7   # ch_0..ch_6
N_BINARY = 12      # ch_7..ch_18

# Semantic masking scenarios where binary channels are excluded
EXCLUDE_BINARY_SCENARIOS: set[str] = {"sleep_gap", "workout_gap", "intensity_failure"}

# Semantic scenarios are excluded from per-category scores
SEMANTIC_SCENARIOS: set[str] = {"sleep_gap", "workout_gap", "intensity_failure"}

# Channel categories for per-category skill scores
CHANNEL_CATEGORIES: dict[str, set[str]] = {
    "activity": {"ch_0", "ch_1", "ch_2", "ch_3", "ch_4"},
    "physiology": {"ch_5", "ch_6"},
    "sleep": {"ch_7", "ch_8"},
    "workouts": {f"ch_{i}" for i in range(9, 19)},
}


def channel_category(channel: str) -> str | None:
    """Return the category name for a channel, or None if uncategorised."""
    for cat, channels in CHANNEL_CATEGORIES.items():
        if channel in channels:
            return cat
    return None


# --------------------------------------------------------------------------
# Error extraction (registry-DataFrame variant)
# --------------------------------------------------------------------------

def extract_errors(
    df: pd.DataFrame,
    split: str,
    subgroup_attr: str = "all",
    subgroup_value: str = "all",
    continuous_metric: str = "nRMSE",
) -> pd.DataFrame:
    """Extract per-task error E for each method from a registry-shaped DF.

    Returns columns ``[method, scenario, channel, channel_type, E]``.
    """
    mask = (
        (df["split"] == split)
        & (df["subgroup_attr"] == subgroup_attr)
        & (df["subgroup_value"] == subgroup_value)
        & (~df["channel"].str.contains("aggregate"))
    )
    sub = df[mask].copy()

    rows = []
    for _, row in sub.iterrows():
        ch_type = row["channel_type"]
        if ch_type == "binary" and row["scenario"] in EXCLUDE_BINARY_SCENARIOS:
            continue
        if ch_type == "continuous":
            e = row.get(continuous_metric)
        elif ch_type == "binary":
            auc = row.get("roc_auc")
            e = 1.0 - auc if pd.notna(auc) else np.nan
        else:
            continue
        if pd.notna(e):
            rows.append({
                "method": row["method"],
                "scenario": row["scenario"],
                "channel": row["channel"],
                "channel_type": ch_type,
                "E": float(e),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Skill score & average rank
# --------------------------------------------------------------------------

def compute_skill_scores(
    errors: pd.DataFrame,
    baseline_errors: pd.DataFrame,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
) -> pd.DataFrame:
    """Skill score per method × scope (overall, scenario, ``cat:<…>``, semantic).

    Skill score:  ``S = 1 − exp(mean(log(clip(E_method / E_baseline))))``.
    """
    bl = baseline_errors.set_index(["scenario", "channel"])["E"]

    ratios = []
    for _, row in errors.iterrows():
        key = (row["scenario"], row["channel"])
        if key not in bl.index:
            continue
        e_baseline = bl[key]
        if e_baseline <= 0 or np.isnan(e_baseline):
            continue
        ratio = row["E"] / e_baseline
        clipped = np.clip(ratio, clip_lower, clip_upper)
        ratios.append({
            "method": row["method"],
            "scenario": row["scenario"],
            "channel": row["channel"],
            "category": channel_category(row["channel"]),
            "clipped_ratio": clipped,
        })

    ratio_df = pd.DataFrame(ratios)
    if ratio_df.empty:
        return pd.DataFrame(columns=["method", "scope", "skill_score", "n_tasks"])

    def _skill(log_ratios: np.ndarray) -> float:
        return 1.0 - float(np.exp(np.mean(log_ratios)))

    results = []
    for (method, scenario), group in ratio_df.groupby(["method", "scenario"], observed=True):
        results.append({
            "method": method, "scope": scenario,
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })
    structural = ratio_df[~ratio_df["scenario"].isin(SEMANTIC_SCENARIOS)]
    for (method, cat), group in structural.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append({
            "method": method, "scope": f"cat:{cat}",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })
    semantic = ratio_df[ratio_df["scenario"].isin(SEMANTIC_SCENARIOS)]
    for method, group in semantic.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "semantic",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })
    for method, group in ratio_df.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "overall",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })
    return pd.DataFrame(results)


def compute_average_rankings(errors: pd.DataFrame) -> pd.DataFrame:
    """Average rank per method × scope. Lower E → better → rank 1."""
    errors = errors.copy()
    errors["rank"] = errors.groupby(["scenario", "channel"], observed=True)["E"].rank(
        method="average", ascending=True,
    )
    errors["category"] = errors["channel"].map(channel_category)

    results = []
    for (method, scenario), group in errors.groupby(["method", "scenario"], observed=True):
        results.append({
            "method": method, "scope": scenario,
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    structural = errors[~errors["scenario"].isin(SEMANTIC_SCENARIOS)]
    for (method, cat), group in structural.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append({
            "method": method, "scope": f"cat:{cat}",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    sem = errors[errors["scenario"].isin(SEMANTIC_SCENARIOS)]
    for method, group in sem.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "semantic",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    for method, group in errors.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "overall",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Disparity-ratio fair skill score
# --------------------------------------------------------------------------

def _per_attribute_skill_keyed(
    df_attr: pd.DataFrame,
    *,
    extra_keys: list[str],
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Per-(method, *extra_keys) fairness skill score for one attribute.

    ``df_attr`` is the long-format error frame for one ``subgroup_attr``
    value (with at least two ``subgroup_value`` levels). Must contain
    ``method, scenario, channel, channel_type, subgroup_value, E`` plus
    every column listed in ``extra_keys`` (e.g. ``"draw"`` for the
    bootstrap path; ``[]`` for the deterministic point estimate).

    For each task r = (scenario, channel) and key tuple k = (*extra_keys):
        D^{(k)}_{r, j}  =  max_g E^{(k)}_{r, j, g}  -  min_g E^{(k)}_{r, j, g}
        D^{(k)}_{r, b}  =  same, for the baseline method b.
        ratio          =  clip(D_j / D_b,  clip_lower, clip_upper)
    Drop tasks where D_b ≤ 0 or any D is NaN. Then
        S^{(k)}_{method}  =  1 - exp(mean_r log(ratio))

    Returns one row per (method, *extra_keys) with columns
    ``[method, *extra_keys, S_attr, n_tasks]``.
    """
    group_keys = [*extra_keys, "method", "scenario", "channel", "channel_type"]
    grouped = df_attr.groupby(group_keys, observed=True)["E"]
    D_max = grouped.max()
    D_min = grouped.min()
    D = (D_max - D_min).rename("D").reset_index()

    # Split baseline rows from model rows on the same task keys (and
    # extra_keys, e.g. draw); merge so each model row carries its paired D_b.
    # Keep the baseline on both sides so its self-ratio (D_b/D_b = 1, clipped
    # → S = 0) lands in the output for parity with compute_skill_scores'
    # treatment of LOCF in skill_scores.csv / skill_scores_bootstrap.csv.
    merge_keys = [*extra_keys, "scenario", "channel", "channel_type"]
    bl = (
        D[D["method"] == baseline_method]
        .drop(columns=["method"])
        .rename(columns={"D": "D_b"})
    )
    jm = D.rename(columns={"D": "D_j"})
    merged = jm.merge(bl, on=merge_keys, how="inner")

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
        return pd.DataFrame(columns=["method", *extra_keys, "S_attr", "n_tasks"])

    ratio = (merged["D_j"] / merged["D_b"]).clip(
        lower=clip_lower, upper=clip_upper,
    )
    merged["log_ratio"] = np.log(ratio.to_numpy())

    agg = (
        merged.groupby(["method", *extra_keys], observed=True)
        .agg(log_ratio_mean=("log_ratio", "mean"), n_tasks=("log_ratio", "size"))
        .reset_index()
    )
    agg["S_attr"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[["method", *extra_keys, "S_attr", "n_tasks"]]


def compute_fair_skill_scores(
    errors: pd.DataFrame,
    *,
    attrs: Iterable[str] = DEFAULT_FAIRNESS_ATTRS,
    baseline_method: str = BASELINE_CONTINUOUS,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
) -> pd.DataFrame:
    """Deterministic disparity-ratio fair skill score per method.

    Input ``errors`` is the long-format per-subgroup error frame:
    ``[method, scenario, channel, channel_type, subgroup_attr,
    subgroup_value, E]``. Rows for the global ``subgroup_attr == "all"``
    cell are ignored.

    For each attribute in ``attrs`` we compute the per-task max-min
    disparity for the method and for the baseline, clip the ratio, and
    take the geometric mean across tasks (see ``_per_attribute_skill_keyed``).
    The overall score is the macro-average across attributes — methods
    missing any attribute drop out of the overall row to keep the average
    honest.

    Returns one row per (method, scope) with columns
    ``[method, scope, fair_skill_score, n_tasks]``. ``scope`` is one entry
    per attribute plus ``"overall"`` for the macro-average.
    """
    attrs = list(attrs)
    per_attr_results: dict[str, pd.DataFrame] = {}
    results: list[dict] = []

    for attr in attrs:
        df_attr = errors[errors["subgroup_attr"] == attr]
        if df_attr.empty:
            continue
        if df_attr["subgroup_value"].nunique() < 2:
            # Max-min disparity is degenerate with a single subgroup.
            continue

        per_attr = _per_attribute_skill_keyed(
            df_attr,
            extra_keys=[],
            baseline_method=baseline_method,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        if per_attr.empty:
            continue
        per_attr_results[attr] = per_attr

        for _, row in per_attr.iterrows():
            results.append({
                "method": row["method"],
                "scope": attr,
                "fair_skill_score": float(row["S_attr"]),
                "n_tasks": int(row["n_tasks"]),
            })

    # Macro-average across attributes per method — drop methods missing
    # any attribute so the mean is honest.
    if per_attr_results:
        stacked = pd.concat(
            [df.assign(attr=name) for name, df in per_attr_results.items()],
            ignore_index=True,
        )
        n_seen = stacked.groupby("method", observed=True)["attr"].nunique()
        full = n_seen[n_seen == len(per_attr_results)].index
        stacked = stacked[stacked["method"].isin(full)]
        overall = (
            stacked.groupby("method", observed=True)
            .agg(S_fair=("S_attr", "mean"), n_tasks=("n_tasks", "sum"))
            .reset_index()
        )
        for _, row in overall.iterrows():
            results.append({
                "method": row["method"],
                "scope": FAIRNESS_OVERALL_SCOPE,
                "fair_skill_score": float(row["S_fair"]),
                "n_tasks": int(row["n_tasks"]),
            })

    if not results:
        return pd.DataFrame(
            columns=["method", "scope", "fair_skill_score", "n_tasks"]
        )
    return pd.DataFrame(results)


def build_baseline_errors(
    errors: pd.DataFrame,
    *,
    baseline_continuous: str = BASELINE_CONTINUOUS,
    baseline_binary: str = BASELINE_BINARY,
) -> pd.DataFrame:
    """Extract global baseline errors (LOCF by default for both channel types).

    Returns one row per ``(scenario, channel)`` with the baseline E.
    """
    cont = errors[
        (errors["method"] == baseline_continuous)
        & (errors["channel_type"] == "continuous")
    ]
    binary = errors[
        (errors["method"] == baseline_binary)
        & (errors["channel_type"] == "binary")
    ]
    return pd.concat([cont, binary], ignore_index=True)
