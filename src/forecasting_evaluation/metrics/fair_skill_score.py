"""Worst-group Fairness Skill Score for forecasting (Track 3).

This is the **default** fairness metric for the forecasting leaderboard. It is
built exactly like the regular forecasting skill score, but evaluated **per
sensitive subgroup** and then reduced by **min over subgroups** (Rawlsian /
worst-case fairness). The reduction guarantees a positive S_fair iff every
eligible subgroup is at least as good as the baseline on a geometric-mean of
per-task ratios.

Formulation. For model ``m``, baseline ``b`` (forecasting: ``seasonal_naive``),
task ``r = (group, metric, channel)``, sensitive attribute ``G ∈ {age_group, sex}``
with subgroup values ``g ∈ G``, ratio clips ``[ℓ, u]``::

    ρ_{r,m,g}  = clip( E_{r,m}^{(g)} / E_{r,b}^{(g)}, ℓ, u )       per (task, subgroup)
    S^{(g)}_m  = 1 − exp( mean_r ln ρ_{r,m,g} )                    per-subgroup skill
    S^{(G)}_m  = min_g S^{(g)}_m                                   worst-subgroup reduction
    S_fair_m   = (1/|A|) · Σ_{G∈A} S^{(G)}_m                       macro-mean across attrs

Notes:
- Subgroups with fewer than ``min_subgroup_users`` users (default 50) are
  dropped at the dataset level — the threshold is fixed once for the cohort
  and applied identically across bootstrap draws.
- A model is only scored for attribute ``G`` if at least 2 eligible subgroups
  contribute a finite ``S^{(g)}_m``; the macro-mean further drops any model
  missing any attribute (consistent with the previous metric's invariant).
- The baseline scores ``S^{(g)}_b = 0`` for every subgroup (self-ratio = 1),
  so ``S_fair = 0`` for the baseline by construction.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics.fairness_skill_score_summary import _task_cols

DEFAULT_FAIRNESS_ATTRS: tuple[str, ...] = ("age_group", "sex")
FAIRNESS_OVERALL_SCOPE = "overall"
CLIP_LOWER = 1e-2
CLIP_UPPER = 100.0
DEFAULT_MIN_SUBGROUP_USERS = 50


def _build_subgroup_error_long(
    error_df: pd.DataFrame,
    demographics: dict[str, dict[str, str]],
    *,
    attrs: Iterable[str] = DEFAULT_FAIRNESS_ATTRS,
) -> pd.DataFrame:
    """Per-user error table -> long per-subgroup task errors.

    ``error_df`` is ``[model, group, metric, channel_idx, channel_name, user_id,
    error]`` (one row per model/task/user). For each attribute, users are mapped
    to a subgroup value (``demographics[user_id][attr]``, default ``"unknown"``)
    and errors are averaged within ``(model, task, subgroup_value)`` to give
    ``E_{r,m}^{(g)}``. Returns ``[model, *task_cols, subgroup_attr,
    subgroup_value, E]``.
    """
    task_cols = _task_cols()
    columns = ["model", *task_cols, "subgroup_attr", "subgroup_value", "E"]
    if error_df.empty:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for attr in attrs:
        subgroup = error_df["user_id"].map(
            lambda uid, a=attr: demographics.get(str(uid), {}).get(a, "unknown")
        )
        tmp = error_df.assign(subgroup_value=subgroup.astype(str))
        grouped = (
            tmp.groupby(["model", *task_cols, "subgroup_value"], observed=True)
            .agg(E=("error", "mean"))
            .reset_index()
        )
        grouped["subgroup_attr"] = attr
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)[columns]


def _eligible_subgroups(
    demographics: dict[str, dict[str, str]],
    user_ids: Iterable[str],
    attrs: Iterable[str],
    min_subgroup_users: int,
) -> dict[str, set[str]]:
    """Subgroup values with ≥ ``min_subgroup_users`` users in the actual cohort.

    Computed once at the dataset level so the same eligibility set is used by
    the deterministic point estimate and every bootstrap draw (bootstrap
    resamples users but does not re-derive eligibility from each draw).
    """
    eligible: dict[str, set[str]] = {}
    user_ids = [str(u) for u in user_ids]
    for attr in attrs:
        counts: dict[str, int] = {}
        for uid in user_ids:
            value = demographics.get(uid, {}).get(attr, "unknown")
            counts[value] = counts.get(value, 0) + 1
        eligible[attr] = {value for value, n in counts.items() if n >= int(min_subgroup_users)}
    return eligible


def _per_subgroup_skill(
    df_attr: pd.DataFrame,
    *,
    extra_keys: list[str],
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Per-(model, subgroup, *extra_keys) skill score over tasks within that subgroup.

    ``df_attr`` is the long per-subgroup error frame for one ``subgroup_attr``
    value. Must contain ``model``, the task columns, ``subgroup_value``, ``E``,
    plus every column in ``extra_keys`` (``"draw"`` for the bootstrap path;
    ``[]`` for the point estimate).

    For each ``(model, subgroup, *extra_keys)``::

        ρ_{r}  = clip(E_m / E_b, clip_lower, clip_upper)   per task r the SAME subgroup
        S_sub  = 1 - exp( mean_r ln ρ_{r} )

    Restricted to the **same (task, subgroup) cell** for model and baseline via
    inner merge — i.e. a model is only credited on a subgroup where the
    baseline also has data.

    Returns one row per ``(model, subgroup_value, *extra_keys)`` with columns
    ``[model, subgroup_value, *extra_keys, S_sub, n_tasks]``.
    """
    task_cols = _task_cols()
    bl_cols = [*extra_keys, *task_cols, "subgroup_value", "E"]
    bl_rows = df_attr.loc[df_attr["model"] == baseline_method, bl_cols].rename(
        columns={"E": "E_b"}
    )
    aligned = df_attr.merge(
        bl_rows, on=[*extra_keys, *task_cols, "subgroup_value"], how="inner"
    )
    if aligned.empty:
        return pd.DataFrame(
            columns=["model", "subgroup_value", *extra_keys, "S_sub", "n_tasks"]
        )

    # Drop non-positive / non-finite errors — log(ratio) requires both > 0.
    finite = (
        aligned["E"].notna()
        & aligned["E_b"].notna()
        & (aligned["E"] > 0)
        & (aligned["E_b"] > 0)
    )
    aligned = aligned.loc[finite].copy()
    if aligned.empty:
        return pd.DataFrame(
            columns=["model", "subgroup_value", *extra_keys, "S_sub", "n_tasks"]
        )

    ratio = (aligned["E"] / aligned["E_b"]).clip(lower=clip_lower, upper=clip_upper)
    aligned["log_ratio"] = np.log(ratio.to_numpy())

    agg = (
        aligned.groupby(["model", "subgroup_value", *extra_keys], observed=True)
        .agg(log_ratio_mean=("log_ratio", "mean"), n_tasks=("log_ratio", "size"))
        .reset_index()
    )
    agg["S_sub"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[["model", "subgroup_value", *extra_keys, "S_sub", "n_tasks"]]


def compute_fair_skill_scores(
    errors: pd.DataFrame,
    *,
    attrs: Iterable[str] = DEFAULT_FAIRNESS_ATTRS,
    baseline_method: str,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
    eligible_subgroups: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Deterministic worst-group fair skill score per model.

    Input ``errors`` is the long per-subgroup error frame from
    ``_build_subgroup_error_long``. ``eligible_subgroups`` is the per-attribute
    set of subgroup values that pass the cohort-level minimum-size threshold
    (built by :func:`_eligible_subgroups`); when ``None`` every subgroup
    present in ``errors`` is eligible.

    Returns one row per ``(model, scope)`` with columns
    ``[model, scope, fair_skill_score, n_tasks]``; ``scope`` is one entry per
    attribute plus ``"overall"`` for the macro-average.
    """
    attrs = list(attrs)
    per_attr_results: dict[str, pd.DataFrame] = {}
    results: list[dict] = []

    for attr in attrs:
        df_attr = errors[errors["subgroup_attr"] == attr]
        if df_attr.empty:
            continue
        if eligible_subgroups is not None:
            allowed = eligible_subgroups.get(attr, set())
            df_attr = df_attr[df_attr["subgroup_value"].isin(allowed)]
            if df_attr.empty:
                continue
        # Need at least 2 distinct eligible subgroups for "worst-group" to be
        # meaningfully an aggregation rather than a single-cell readout.
        if df_attr["subgroup_value"].nunique() < 2:
            continue

        per_sub = _per_subgroup_skill(
            df_attr,
            extra_keys=[],
            baseline_method=baseline_method,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        if per_sub.empty:
            continue

        # Worst-subgroup reduction per model — require ≥ 2 subgroups present so
        # a model that happens to have data on only one subgroup is not silently
        # credited with a single-cell "worst" score.
        per_attr = (
            per_sub.groupby("model", observed=True)
            .agg(
                S_attr=("S_sub", "min"),
                n_tasks=("n_tasks", "sum"),
                n_subgroups=("subgroup_value", "nunique"),
            )
            .reset_index()
        )
        per_attr = per_attr[per_attr["n_subgroups"] >= 2]
        if per_attr.empty:
            continue
        per_attr_results[attr] = per_attr

        for _, row in per_attr.iterrows():
            results.append(
                {
                    "model": row["model"],
                    "scope": attr,
                    "fair_skill_score": float(row["S_attr"]),
                    "n_tasks": int(row["n_tasks"]),
                }
            )

    # Macro-average across attributes per model — drop models missing any
    # attribute so the mean stays honest.
    if per_attr_results:
        stacked = pd.concat(
            [df.assign(attr=name) for name, df in per_attr_results.items()],
            ignore_index=True,
        )
        n_seen = stacked.groupby("model", observed=True)["attr"].nunique()
        full = n_seen[n_seen == len(per_attr_results)].index
        stacked = stacked[stacked["model"].isin(full)]
        overall = (
            stacked.groupby("model", observed=True)
            .agg(S_fair=("S_attr", "mean"), n_tasks=("n_tasks", "sum"))
            .reset_index()
        )
        for _, row in overall.iterrows():
            results.append(
                {
                    "model": row["model"],
                    "scope": FAIRNESS_OVERALL_SCOPE,
                    "fair_skill_score": float(row["S_fair"]),
                    "n_tasks": int(row["n_tasks"]),
                }
            )

    if not results:
        return pd.DataFrame(columns=["model", "scope", "fair_skill_score", "n_tasks"])
    return pd.DataFrame(results)


def compute_fair_skill_scores_from_errors(
    error_df: pd.DataFrame,
    demographics: dict[str, dict[str, str]],
    *,
    attrs: Iterable[str] = DEFAULT_FAIRNESS_ATTRS,
    baseline_method: str,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
    min_subgroup_users: int = DEFAULT_MIN_SUBGROUP_USERS,
) -> pd.DataFrame:
    """Point estimate straight from the per-user ``error_df`` + demographics.

    Convenience wrapper: builds the long per-subgroup frame, derives the
    cohort-level subgroup eligibility set, then calls
    :func:`compute_fair_skill_scores`. Shared by the bootstrap (point/identity
    draw) and the paper pipeline's Phase-2 deterministic CSV.
    """
    long = _build_subgroup_error_long(error_df, demographics, attrs=attrs)
    eligible = _eligible_subgroups(
        demographics,
        set(error_df["user_id"].astype(str).tolist()),
        attrs,
        min_subgroup_users,
    )
    return compute_fair_skill_scores(
        long,
        attrs=attrs,
        baseline_method=baseline_method,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        eligible_subgroups=eligible,
    )
