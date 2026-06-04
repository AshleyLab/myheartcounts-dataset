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
