"""Disparity-ratio Fairness Skill Score for forecasting (Track 3).

This is the **default** fairness metric (replacing the legacy ``S − λ·D``
"fairness-adjusted skill score" in ``fairness_skill_score_summary.py``). It is
built exactly like the regular forecasting skill score, but the quantity scored
per task is the **cross-subgroup error gap**, taken as a **ratio against the
baseline's gap** (no ``λ``). It mirrors the imputation track's metric in
``imputation_evaluation/evaluation/paper_metrics_core.py`` (``compute_fair_skill_scores``),
adapted to forecasting's per-``(channel, metric)`` tasks and made
**category-balanced**: the per-task log-ratios collapse to one value per sensor
scope (activity / physiology / sleep / workout) before averaging, so the 10
workout channels can't dominate. It additionally emits per-scope and per-channel
breakdowns. Kept here (not imported from the imputation internals) so the public
forecasting package stays decoupled — same reason the bootstrap helpers were
copied in ``bootstrap_skill_rank.py``.

Formulation. For model ``m``, baseline ``b`` (forecasting: ``seasonal_naive``),
task ``r = (group, metric, channel)``, sensitive attribute ``G ∈ {age_group, sex}``
with subgroup values ``g`` (the ``unknown`` bucket is a real subgroup, kept),
ratio clips ``[ℓ, u]``::

    D_{r,m}^{(G)} = (2 / |G|(|G|-1)) · Σ_{g≠g'} | E_{r,m}^{(g)} − E_{r,m}^{(g')} |   (same for b)
    the mean absolute pairwise difference (MAPD) over the common method∩baseline
    subgroup set per task; for |G| = 2 this collapses to |E_a − E_b| (== max-min)
    drop task r from G if <2 common subgroups, D_{r,b}^{(G)} ≤ 0, or any D is NaN
    ρ_r          = clip( D_{r,m}^{(G)} / D_{r,b}^{(G)}, ℓ, u )
    S^{(G)}_m    = 1 − exp( mean_c [ mean_{r∈c} ln ρ_r ] )   (c = sensor scope; equal per scope)
    S_fair_m     = (1/|A|) · Σ_{G∈A} S^{(G)}_m              (macro-average across attrs)

Per-scope rows use the inner ``1 − exp(mean_{r∈c} ln ρ_r)`` (one scope c); per-
channel rows use ``1 − exp`` over that channel's metric tasks. The headline scopes
(``overall``, the 4 categories, ``channel_<i>``) are each macro-averaged across
attributes, dropping any model/key missing an attribute to keep the mean honest;
the baseline's self-ratio is ``1`` (⇒ ``S_b = 0``).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics import metric_spec as _spec
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


def _mapd(series: pd.Series) -> float:
    """Mean absolute pairwise difference of a numeric series.

    Computes ``(2 / n(n-1)) * Σ_{i<j} |x_i − x_j|`` — the mean of ``|x_i − x_j|``
    over the ``n(n-1)/2`` unordered pairs. Returns NaN when fewer than 2 finite
    values are present. For ``n = 2`` this is ``|x_0 − x_1|`` (== max-min); for
    ``n ≥ 3`` it smooths over every pair instead of only the two extremes.
    """
    vals = series.to_numpy()
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n < 2:
        return float("nan")
    diffs = np.abs(vals[:, None] - vals[None, :])
    iu, ju = np.triu_indices(n, k=1)
    return float(diffs[iu, ju].mean())


def _per_task_disparity_log_ratio(
    df_attr: pd.DataFrame,
    *,
    baseline_method: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Per-task disparity log-ratios for one attribute (pre-aggregation).

    ``df_attr`` is the long per-subgroup error frame for one ``subgroup_attr``
    value (columns ``model``, the task columns, ``subgroup_value``, ``E``). For each
    task ``r = (group, metric, channel)`` we restrict to the **common subgroup set**
    that both the model and the baseline have data for, then::

        D_j = mean absolute pairwise difference of E_j over common subgroups
        D_b = same, for the baseline b
        ratio = clip(D_j / D_b, clip_lower, clip_upper)

    Drop tasks with fewer than two common subgroups, where the baseline is already
    perfectly fair (``D_b <= 0``), or where any D is NaN. The sensor-category
    ``scope`` (activity / physiology / sleep / workout) is attached per task.

    The >=2-common-subgroup guard prevents a known failure mode where a model that
    happens to have data for only one subgroup of a task/draw would yield
    ``D_j = 0`` by construction (a single value has no pairwise differences), get
    clipped to ``clip_lower`` after dividing by ``D_b > 0``, and earn a near-perfect
    score for free. This can happen in the bootstrap path (per-draw row drop-outs)
    even when the upstream subgroup universe is logically the same across models.

    Returns one row per surviving task with columns
    ``[model, group, metric, channel_idx, channel_name, scope, log_ratio]``.
    """
    task_cols = _task_cols()
    model_task_keys = [*task_cols, "model"]
    out_cols = ["model", *task_cols, "scope", "log_ratio"]

    # Pair each model row with the baseline's E for the same (task, subgroup_value).
    # The inner merge restricts every (model, task) row set to subgroups the baseline
    # also has data for, so D_j and D_b are computed over the SAME subgroup set per
    # task, and excludes orphan rows that would collapse D_j to 0 when a model has
    # only one subgroup row for a task/draw.
    bl_rows = df_attr.loc[
        df_attr["model"] == baseline_method,
        [*task_cols, "subgroup_value", "E"],
    ].rename(columns={"E": "E_b"})
    aligned = df_attr.merge(bl_rows, on=[*task_cols, "subgroup_value"], how="inner")
    if aligned.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = aligned.groupby(model_task_keys, observed=True)
    D = pd.DataFrame(
        {
            "D_j": grouped["E"].apply(_mapd),
            "D_b": grouped["E_b"].apply(_mapd),
            "n_sub": grouped["subgroup_value"].nunique(),
        }
    ).reset_index()

    keep = (
        (D["n_sub"] >= 2) & (D["D_b"] > 0) & D["D_b"].notna() & D["D_j"].notna() & (D["D_j"] >= 0)
    )
    D = D.loc[keep].copy()
    if D.empty:
        return pd.DataFrame(columns=out_cols)

    ratio = (D["D_j"] / D["D_b"]).clip(lower=clip_lower, upper=clip_upper)
    D["log_ratio"] = np.log(ratio.to_numpy())
    D["scope"] = D["channel_idx"].map(_spec.category_scope_for_channel)
    return D[out_cols]


def _category_balanced_skill(tasks: pd.DataFrame) -> pd.DataFrame:
    """Two-stage category-balanced fairness skill per model.

    Stage 1: mean ``log_ratio`` within each sensor scope. Stage 2: equal mean across
    the scopes present, ``fair_skill_score = 1 - exp(stage2)``. ``n_tasks`` is the
    number of scopes present (<=4). Returns ``[model, fair_skill_score, n_tasks]``.
    """
    cols = ["model", "fair_skill_score", "n_tasks"]
    valid = tasks[tasks["scope"].notna()]
    if valid.empty:
        return pd.DataFrame(columns=cols)
    stage1 = (
        valid.groupby(["model", "scope"], observed=True)["log_ratio"]
        .mean()
        .reset_index(name="scope_log")
    )
    agg = (
        stage1.groupby("model", observed=True)
        .agg(log_ratio_mean=("scope_log", "mean"), n_tasks=("scope_log", "size"))
        .reset_index()
    )
    agg["fair_skill_score"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[cols]


def _skill_by(tasks: pd.DataFrame, *, key_cols: list[str]) -> pd.DataFrame:
    """Single-stage fairness skill at a given granularity.

    ``1 - exp(mean log_ratio)`` over the tasks in each ``key_cols`` group. Returns
    ``[*key_cols, fair_skill_score, n_tasks]`` (groups with a NaN key are dropped by
    ``groupby``).
    """
    cols = [*key_cols, "fair_skill_score", "n_tasks"]
    if tasks.empty:
        return pd.DataFrame(columns=cols)
    agg = (
        tasks.groupby(key_cols, observed=True)
        .agg(log_ratio_mean=("log_ratio", "mean"), n_tasks=("log_ratio", "size"))
        .reset_index()
    )
    agg["fair_skill_score"] = 1.0 - np.exp(agg["log_ratio_mean"])
    return agg[cols]


def _macro_across_attrs(per_attr: dict[str, pd.DataFrame], *, key_cols: list[str]) -> pd.DataFrame:
    """Macro-average ``fair_skill_score`` across attributes per ``key_cols`` tuple.

    Drops any key tuple not present in EVERY attribute (the honesty rule used for the
    ``overall`` row, applied at each granularity). ``n_tasks`` sums across attributes.
    Returns ``[*key_cols, fair_skill_score, n_tasks]``.
    """
    cols = [*key_cols, "fair_skill_score", "n_tasks"]
    n_attrs = len(per_attr)
    if n_attrs == 0:
        return pd.DataFrame(columns=cols)
    stacked = pd.concat([df.assign(_attr=name) for name, df in per_attr.items()], ignore_index=True)
    if stacked.empty:
        return pd.DataFrame(columns=cols)
    seen = stacked.groupby(key_cols, observed=True)["_attr"].transform("nunique")
    stacked = stacked[seen == n_attrs]
    if stacked.empty:
        return pd.DataFrame(columns=cols)
    agg = (
        stacked.groupby(key_cols, observed=True)
        .agg(fair_skill_score=("fair_skill_score", "mean"), n_tasks=("n_tasks", "sum"))
        .reset_index()
    )
    return agg[cols]


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
    columns ``[model, scope, fair_skill_score, n_tasks]``. ``scope`` is:

    * one entry per attribute (``age_group`` / ``sex``) — the **category-balanced**
      per-attribute skill;
    * ``overall`` — macro-average of the per-attribute skills;
    * the 4 sensor categories (``activity`` / ``physiology`` / ``sleep`` /
      ``workout``) and the per-channel ``channel_<i>`` rows — each macro-averaged
      across attributes.

    Per-attribute scores collapse to one value per sensor scope before averaging, so
    the 10 workout channels can't dominate. The macro rows drop any model/key missing
    from an attribute, keeping the average honest.
    """
    attrs = list(attrs)
    per_attr_overall: dict[str, pd.DataFrame] = {}
    per_attr_scope: dict[str, pd.DataFrame] = {}
    per_attr_channel: dict[str, pd.DataFrame] = {}
    results: list[dict] = []

    for attr in attrs:
        df_attr = errors[errors["subgroup_attr"] == attr]
        if df_attr.empty:
            continue
        if df_attr["subgroup_value"].nunique() < 2:
            # MAPD disparity is undefined with a single subgroup (no pairs).
            continue

        tasks = _per_task_disparity_log_ratio(
            df_attr,
            baseline_method=baseline_method,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        if tasks.empty:
            continue
        attr_skill = _category_balanced_skill(tasks)
        if attr_skill.empty:
            continue
        per_attr_overall[attr] = attr_skill
        per_attr_scope[attr] = _skill_by(tasks, key_cols=["model", "scope"])
        per_attr_channel[attr] = _skill_by(tasks, key_cols=["model", "channel_idx"])

        for _, row in attr_skill.iterrows():
            results.append(
                {
                    "model": row["model"],
                    "scope": attr,
                    "fair_skill_score": float(row["fair_skill_score"]),
                    "n_tasks": int(row["n_tasks"]),
                }
            )

    # Macro-average across attributes at each granularity (overall / per-scope /
    # per-channel), dropping any model[/key] missing an attribute.
    for _, row in _macro_across_attrs(per_attr_overall, key_cols=["model"]).iterrows():
        results.append(
            {
                "model": row["model"],
                "scope": FAIRNESS_OVERALL_SCOPE,
                "fair_skill_score": float(row["fair_skill_score"]),
                "n_tasks": int(row["n_tasks"]),
            }
        )
    for _, row in _macro_across_attrs(per_attr_scope, key_cols=["model", "scope"]).iterrows():
        results.append(
            {
                "model": row["model"],
                "scope": str(row["scope"]),
                "fair_skill_score": float(row["fair_skill_score"]),
                "n_tasks": int(row["n_tasks"]),
            }
        )
    for _, row in _macro_across_attrs(
        per_attr_channel, key_cols=["model", "channel_idx"]
    ).iterrows():
        results.append(
            {
                "model": row["model"],
                "scope": f"channel_{int(row['channel_idx'])}",
                "fair_skill_score": float(row["fair_skill_score"]),
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
