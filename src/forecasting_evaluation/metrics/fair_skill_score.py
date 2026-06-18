"""Disparity-ratio Fairness Skill Score for forecasting (Track 3).

This is the **default** fairness metric (replacing the legacy ``S − λ·D``
"fairness-adjusted skill score" in ``fairness_skill_score_summary.py``). It is
built exactly like the regular forecasting skill score, but the quantity scored
per task is the **cross-subgroup error gap**, taken as a **ratio against the
baseline's gap** (no ``λ``). It mirrors the imputation track's metric in
``imputation_evaluation/evaluation/paper_metrics_core.py`` (``compute_fair_skill_scores`` /
``_per_attribute_skill_keyed``); the two core functions are ported here verbatim
(renamed ``method→model`` and keyed on forecasting task columns) so the public
forecasting package stays decoupled from the imputation internals — same reason
the bootstrap helpers were copied in ``bootstrap_skill_rank.py``.

Formulation. For model ``m``, baseline ``b`` (forecasting: ``seasonal_naive``),
task ``r = (group, metric, channel)``, sensitive attribute ``G ∈ {age_group, sex}``
with subgroup values ``g`` (the ``unknown`` bucket is a real subgroup, kept),
ratio clips ``[ℓ, u]``::

    D_{r,m}^{(G)} = max_g E_{r,m}^{(g)} − min_g E_{r,m}^{(g)}     (same for b)
    both gaps are taken over the common method∩baseline subgroup set per task
    drop task r from G if <2 common subgroups, D_{r,b}^{(G)} ≤ 0, or any D is NaN
    ρ_r          = clip( D_{r,m}^{(G)} / D_{r,b}^{(G)}, ℓ, u )
    S^{(G)}_m    = 1 − exp( mean_r ln ρ_r )
    S_fair_m     = (1/|A|) · Σ_{G∈A} S^{(G)}_m     (macro-average across attrs)

The baseline's self-ratio is ``1`` (⇒ ``S_b = 0``), and a model missing any
attribute drops out of the ``overall`` row to keep the macro-average honest.
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


def _per_attribute_skill_keyed(
    df_attr: pd.DataFrame,
    *,
    extra_keys: list[str],
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Per-(model, *extra_keys) fairness skill score for one attribute.

    ``df_attr`` is the long per-subgroup error frame for one ``subgroup_attr``
    value. Must contain ``model``, the task columns, ``subgroup_value``, ``E``,
    plus every column in ``extra_keys`` (``"draw"`` for the bootstrap path;
    ``[]`` for the point estimate).

    For each task ``r`` and key tuple ``k = (*extra_keys)``, we restrict to the
    **common subgroup set** that both the model and the baseline have data for
    in that task/key, and then::

        D_j = max_g E_j^{(g)} - min_g E_j^{(g)}     (over common subgroups)
        D_b = same, for the baseline b
        ratio = clip(D_j / D_b, clip_lower, clip_upper)

    Drop tasks where fewer than two common subgroups exist, where the baseline
    is already perfectly fair (D_b <= 0), or where any D is NaN. Then
    ``S_attr = 1 - exp(mean_r log(ratio))``.

    The >=2-common-subgroup guard prevents a known failure mode where a model
    that happens to have data for only one subgroup of a task/draw would yield
    ``D_j = max - min = 0`` by construction, get clipped to ``clip_lower`` after
    dividing by ``D_b > 0``, and earn a near-perfect ``S_attr ~= 1 - clip_lower``
    for free. This can happen in the bootstrap path (per-draw row drop-outs from
    non-finite metrics, missing manifest coverage) even when the upstream
    subgroup universe is logically the same across models.

    Returns one row per ``(model, *extra_keys)`` with columns
    ``[model, *extra_keys, S_attr, n_tasks]``.
    """
    task_cols = _task_cols()
    task_keys = [*extra_keys, *task_cols]
    model_task_keys = [*task_keys, "model"]

    # Pair each model row with the baseline's E for the same (task,
    # subgroup_value). The inner merge restricts every (model, task) row set to
    # subgroups the baseline also has data for, so D_j and D_b are computed over
    # the SAME subgroup set per task, and excludes orphan rows that would
    # collapse D_j to 0 when a model has only one subgroup row for a task/draw.
    bl_rows = df_attr.loc[
        df_attr["model"] == baseline_method,
        [*task_keys, "subgroup_value", "E"],
    ].rename(columns={"E": "E_b"})
    aligned = df_attr.merge(bl_rows, on=[*task_keys, "subgroup_value"], how="inner")
    if aligned.empty:
        return pd.DataFrame(columns=["model", *extra_keys, "S_attr", "n_tasks"])

    grouped = aligned.groupby(model_task_keys, observed=True)
    D = pd.DataFrame(
        {
            "D_j": grouped["E"].max() - grouped["E"].min(),
            "D_b": grouped["E_b"].max() - grouped["E_b"].min(),
            "n_sub": grouped["subgroup_value"].nunique(),
        }
    ).reset_index()

    # Drop tasks with <2 common (model ∩ baseline) subgroups (max-min is
    # degenerate and would be rewarded as "perfect fairness" after clipping),
    # where the baseline is already perfectly fair (D_b <= 0), or where any D is
    # NaN. max-min is non-negative by construction.
    keep = (
        (D["n_sub"] >= 2) & (D["D_b"] > 0) & D["D_b"].notna() & D["D_j"].notna() & (D["D_j"] >= 0)
    )
    D = D.loc[keep].copy()
    if D.empty:
        return pd.DataFrame(columns=["model", *extra_keys, "S_attr", "n_tasks"])

    ratio = (D["D_j"] / D["D_b"]).clip(lower=clip_lower, upper=clip_upper)
    D["log_ratio"] = np.log(ratio.to_numpy())

    agg = (
        D.groupby(["model", *extra_keys], observed=True)
        .agg(log_ratio_mean=("log_ratio", "mean"), n_tasks=("log_ratio", "size"))
        .reset_index()
    )
    agg["S_attr"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[["model", *extra_keys, "S_attr", "n_tasks"]]


def compute_fair_skill_scores(
    errors: pd.DataFrame,
    *,
    attrs: Iterable[str] = DEFAULT_FAIRNESS_ATTRS,
    baseline_method: str,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
) -> pd.DataFrame:
    """Deterministic disparity-ratio fair skill score per model.

    Input ``errors`` is the long per-subgroup error frame from
    ``_build_subgroup_error_long``. Returns one row per ``(model, scope)`` with
    columns ``[model, scope, fair_skill_score, n_tasks]``; ``scope`` is one entry
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
            # max-min disparity is degenerate with a single subgroup.
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
) -> pd.DataFrame:
    """Point estimate straight from the per-user ``error_df`` + demographics.

    Thin convenience wrapper: builds the long per-subgroup frame then calls
    ``compute_fair_skill_scores``. Shared by the bootstrap (point/identity draw)
    and the paper pipeline's Phase-2 deterministic CSV.
    """
    long = _build_subgroup_error_long(error_df, demographics, attrs=attrs)
    return compute_fair_skill_scores(
        long,
        attrs=attrs,
        baseline_method=baseline_method,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
