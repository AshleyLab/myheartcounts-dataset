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

# Binary categories eligible for the collapsed-scope leaderboard rows.
# Each entry maps a category name to its ordered tuple of channel indices.
# Order matters for stable column layout in the precomputed per-(user, cat)
# matrix that drives the per-draw E for ``cat_collapsed:*`` and
# ``overall_binary_collapsed`` scopes — see ``bootstrap_skill_rank.py``'s
# ``_per_method_cell_collapsed_errors``. Continuous categories are
# deliberately NOT collapsed (decided in plan Part D): the unequal channel
# counts that motivate the binary collapse don't apply to continuous,
# whose seven channels already weight roughly fairly under the per-task
# geomean.
BINARY_CATEGORIES_ORDERED: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("sleep",    (7, 8)),
    ("workouts", tuple(range(9, 19))),
)


def channel_category(channel: str) -> str | None:
    """Return the category name for a channel, or None if uncategorised."""
    for cat, channels in CHANNEL_CATEGORIES.items():
        if channel in channels:
            return cat
    return None


def is_collapsed_channel(channel: str) -> bool:
    """True for the synthetic ``cat_collapsed:*`` channel labels emitted by Part D."""
    return channel.startswith("cat_collapsed:")


# --------------------------------------------------------------------------
# Error extraction (registry-DataFrame variant)
# --------------------------------------------------------------------------

def extract_errors(
    df: pd.DataFrame,
    split: str,
    subgroup_attr: str = "all",
    subgroup_value: str = "all",
    continuous_metric: str = "RMSE",
) -> pd.DataFrame:
    """Extract per-task error E for each method from a registry-shaped DF.

    Returns columns ``[method, scenario, channel, channel_type, E]``.

    ``continuous_metric`` defaults to ``"RMSE"`` (changed from ``"nRMSE"`` in
    the user-macro refactor). At per-channel scope the per-task skill ratio
    ``E_method / E_baseline`` is identical under RMSE and nRMSE — the
    ``channel_std`` factor cancels exactly — so the skill score is unchanged.
    Switching to RMSE keeps the leaderboard absolute-value columns in their
    natural units. Callers may still pass ``continuous_metric="nRMSE"`` to
    read the legacy column if their registry DataFrame carries both.
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
    """Skill score per method × scope.

    Emits these scopes:

    Per-channel (the historical layout):
      - ``<scenario>`` (one per scenario)
      - ``cat:activity`` / ``cat:physiology`` / ``cat:sleep`` / ``cat:workouts``
        (structural scenarios only)
      - ``semantic`` (semantic scenarios only)
      - ``overall`` (all 19 per-channel tasks × all scenarios)

    Collapsed-binary (added by Part D — leaderboard reports side-by-side):
      - ``cat_collapsed:sleep`` / ``cat_collapsed:workouts`` (one geomean per
        category × scenario task set)
      - ``overall_binary_collapsed`` (7 continuous per-channel tasks + 2
        binary categories × scenarios — same task layout as ``overall``
        with the 12 binary channels replaced by 2 collapsed-category tasks)

    Skill score: ``S = 1 − exp(mean(log(clip(E_method / E_baseline))))``.

    The collapsed scopes consume rows where ``channel`` starts with
    ``cat_collapsed:`` (produced upstream by
    ``bootstrap_skill_rank._per_method_cell_collapsed_errors``). The legacy
    per-channel scopes explicitly exclude those rows so they don't
    double-count the binary information.
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
            "is_collapsed": is_collapsed_channel(row["channel"]),
        })

    ratio_df = pd.DataFrame(ratios)
    if ratio_df.empty:
        return pd.DataFrame(columns=["method", "scope", "skill_score", "n_tasks"])

    def _skill(log_ratios: np.ndarray) -> float:
        return 1.0 - float(np.exp(np.mean(log_ratios)))

    # Partition into legacy per-channel rows vs collapsed-binary rows.
    legacy_df = ratio_df[~ratio_df["is_collapsed"]]
    collapsed_df = ratio_df[ratio_df["is_collapsed"]]

    results = []

    # --- Per-scenario (per-channel) -----------------------------------
    for (method, scenario), group in legacy_df.groupby(["method", "scenario"], observed=True):
        results.append({
            "method": method, "scope": scenario,
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })

    # --- cat:<cat> (structural scenarios only, per-channel) -----------
    structural = legacy_df[~legacy_df["scenario"].isin(SEMANTIC_SCENARIOS)]
    for (method, cat), group in structural.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append({
            "method": method, "scope": f"cat:{cat}",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })

    # --- semantic (semantic scenarios only, per-channel) --------------
    semantic = legacy_df[legacy_df["scenario"].isin(SEMANTIC_SCENARIOS)]
    for method, group in semantic.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "semantic",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })

    # --- overall (per-channel; historical leaderboard column) ----------
    for method, group in legacy_df.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "overall",
            "skill_score": _skill(np.log(group["clipped_ratio"].values)),
            "n_tasks": len(group),
        })

    # --- cat_collapsed:<cat> (one per binary category) ----------------
    if not collapsed_df.empty:
        for (method, channel), group in collapsed_df.groupby(["method", "channel"], observed=True):
            cat_name = str(channel).split(":", 1)[1]
            results.append({
                "method": method, "scope": f"cat_collapsed:{cat_name}",
                "skill_score": _skill(np.log(group["clipped_ratio"].values)),
                "n_tasks": len(group),
            })

        # --- overall_binary_collapsed: continuous per-channel tasks +
        #     collapsed-binary tasks (the binary side of "overall" replaced).
        #     Excludes per-channel binary rows so each binary task is counted
        #     once.
        continuous_legacy = legacy_df[legacy_df["category"].isin({"activity", "physiology"})]
        merged = pd.concat([continuous_legacy, collapsed_df], ignore_index=True)
        for method, group in merged.groupby("method", observed=True):
            results.append({
                "method": method, "scope": "overall_binary_collapsed",
                "skill_score": _skill(np.log(group["clipped_ratio"].values)),
                "n_tasks": len(group),
            })

    return pd.DataFrame(results)


def compute_average_rankings(errors: pd.DataFrame) -> pd.DataFrame:
    """Average rank per method × scope. Lower E → better → rank 1.

    Emits the same scope set as :func:`compute_skill_scores`: per-channel
    legacy scopes (``<scenario>``, ``cat:*``, ``semantic``, ``overall``)
    plus the Part D collapsed-binary scopes (``cat_collapsed:sleep``,
    ``cat_collapsed:workouts``, ``overall_binary_collapsed``) when
    ``cat_collapsed:*`` rows are present.

    Ranks are computed within each ``(scenario, channel)`` task across
    methods, so collapsed-binary tasks rank methods against each other
    inside the collapsed scope without bleeding into per-channel rank
    computations.
    """
    errors = errors.copy()
    errors["rank"] = errors.groupby(["scenario", "channel"], observed=True)["E"].rank(
        method="average", ascending=True,
    )
    errors["category"] = errors["channel"].map(channel_category)
    errors["is_collapsed"] = errors["channel"].map(is_collapsed_channel)

    legacy_errors = errors[~errors["is_collapsed"]]
    collapsed_errors = errors[errors["is_collapsed"]]

    results = []
    for (method, scenario), group in legacy_errors.groupby(["method", "scenario"], observed=True):
        results.append({
            "method": method, "scope": scenario,
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    structural = legacy_errors[~legacy_errors["scenario"].isin(SEMANTIC_SCENARIOS)]
    for (method, cat), group in structural.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append({
            "method": method, "scope": f"cat:{cat}",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    sem = legacy_errors[legacy_errors["scenario"].isin(SEMANTIC_SCENARIOS)]
    for method, group in sem.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "semantic",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })
    for method, group in legacy_errors.groupby("method", observed=True):
        results.append({
            "method": method, "scope": "overall",
            "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
        })

    # --- Part D scopes (collapsed binary categories) ------------------
    if not collapsed_errors.empty:
        for (method, channel), group in collapsed_errors.groupby(["method", "channel"], observed=True):
            cat_name = str(channel).split(":", 1)[1]
            results.append({
                "method": method, "scope": f"cat_collapsed:{cat_name}",
                "avg_rank": float(group["rank"].mean()), "n_tasks": len(group),
            })
        continuous_legacy = legacy_errors[
            legacy_errors["category"].isin({"activity", "physiology"})
        ]
        merged = pd.concat([continuous_legacy, collapsed_errors], ignore_index=True)
        for method, group in merged.groupby("method", observed=True):
            results.append({
                "method": method, "scope": "overall_binary_collapsed",
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

    For each task r = (scenario, channel) and key tuple k = (*extra_keys),
    we restrict to the **common subgroup set** that both the method and the
    baseline have data for in that task/key, and then:
        D^{(k)}_{r, j}  =  max_g E^{(k)}_{r, j, g}  -  min_g E^{(k)}_{r, j, g}
        D^{(k)}_{r, b}  =  same, for the baseline method b.
        ratio          =  clip(D_j / D_b,  clip_lower, clip_upper)
    Drop tasks where fewer than two common subgroups exist, where the
    baseline is already perfectly fair (D_b ≤ 0), or where any D is NaN.
    Then
        S^{(k)}_{method}  =  1 - exp(mean_r log(ratio))

    The ≥2-common-subgroup guard prevents a known failure mode where a
    method that happens to have data for only one subgroup of a task/draw
    would yield D_j = max - min = 0 by construction, get clipped to
    ``clip_lower`` after dividing by D_b > 0, and earn a near-perfect
    ``S_attr ≈ 1 - clip_lower`` for free. This can happen in the bootstrap
    path (per-draw row drop-outs from non-finite metrics, single-class AUC
    draws, missing manifest coverage) even when the upstream subgroup
    universe is logically the same across methods.

    Returns one row per (method, *extra_keys) with columns
    ``[method, *extra_keys, S_attr, n_tasks]``.
    """
    task_keys = [*extra_keys, "scenario", "channel", "channel_type"]
    method_task_keys = [*task_keys, "method"]

    # Pair each method row with the baseline's E for the same (task,
    # subgroup_value). The inner merge restricts every (method, task) row
    # set to subgroups the baseline also has data for, so D_j and D_b are
    # computed over the SAME subgroup set per task. This both aligns the
    # comparison and excludes orphan rows that would collapse D_j to 0 when
    # a method has only one subgroup row for a task/draw.
    bl_rows = (
        df_attr.loc[
            df_attr["method"] == baseline_method,
            [*task_keys, "subgroup_value", "E"],
        ]
        .rename(columns={"E": "E_b"})
    )
    aligned = df_attr.merge(
        bl_rows, on=[*task_keys, "subgroup_value"], how="inner",
    )
    if aligned.empty:
        return pd.DataFrame(columns=["method", *extra_keys, "S_attr", "n_tasks"])

    grouped = aligned.groupby(method_task_keys, observed=True)
    D = pd.DataFrame({
        "D_j": grouped["E"].max() - grouped["E"].min(),
        "D_b": grouped["E_b"].max() - grouped["E_b"].min(),
        "n_sub": grouped["subgroup_value"].nunique(),
    }).reset_index()

    # Drop tasks where:
    #   - fewer than 2 common (method ∩ baseline) subgroups exist; max-min
    #     is degenerate (= 0) by construction and would otherwise be
    #     rewarded as "perfect fairness" after clipping.
    #   - the baseline is already perfectly fair (D_b ≤ 0).
    #   - either disparity is NaN.
    # The D_b > 0 / NaN guards mirror compute_skill_scores'
    # ``e_baseline <= 0 or np.isnan(e_baseline)`` drop rule.
    keep = (
        (D["n_sub"] >= 2)
        & (D["D_b"] > 0)
        & D["D_b"].notna()
        & D["D_j"].notna()
        & (D["D_j"] >= 0)  # max-min is non-negative by construction
    )
    D = D.loc[keep].copy()
    if D.empty:
        return pd.DataFrame(columns=["method", *extra_keys, "S_attr", "n_tasks"])

    ratio = (D["D_j"] / D["D_b"]).clip(lower=clip_lower, upper=clip_upper)
    D["log_ratio"] = np.log(ratio.to_numpy())

    agg = (
        D.groupby(["method", *extra_keys], observed=True)
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
