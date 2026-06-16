"""Paired participant-level (user) bootstrap for the forecasting fair skill score.

Wraps the deterministic disparity-ratio metric in ``fair_skill_score.py`` with
the same paired user-bootstrap used for skill/rank (``bootstrap_skill_rank.py``):
one shared user-resample matrix, the per-user error table built **once**, and the
point-flow recomputed per draw on the replica-expanded table. The cluster unit is
the **user**, so between-user variance is captured, and the within-draw pairing of
the model gap ``D_m`` against the baseline gap ``D_b`` is preserved (both come from
one resampled cohort). The identity resample reproduces the deterministic point
estimate — see ``tests/test_forecasting_fair_skill_score_bootstrap.py``.

Both estimates are returned: ``fairness_skill_scores`` (bootstrap CIs) and
``fairness_skill_scores_point`` (deterministic point), built from the same
``error_df`` so they are guaranteed consistent.
"""

from __future__ import annotations

import logging

import pandas as pd

from forecasting_evaluation.metrics.bootstrap_skill_rank import (
    _bootstrap_indices,
    _draw_replica_frame,
    _resample,
    _seed_for,
    _summary_table,
)
from forecasting_evaluation.metrics.fair_skill_score import (
    CLIP_LOWER,
    CLIP_UPPER,
    DEFAULT_FAIRNESS_ATTRS,
    _build_subgroup_error_long,
    compute_fair_skill_scores,
    compute_fair_skill_scores_from_errors,
)
from forecasting_evaluation.metrics.fairness_skill_score_summary import (
    DEFAULT_AGE_BINS,
    _build_error_table,
    load_user_demographics,
)

logger = logging.getLogger(__name__)

_SUMMARY_COLUMNS = ["model", "scope", "mean", "se", "ci_lo", "ci_hi", "n_boot"]


def bootstrap_fair_skill_score(
    *,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_channel_indices: tuple[int, ...],
    attrs: tuple[str, ...] = DEFAULT_FAIRNESS_ATTRS,
    demographics: dict[str, dict[str, str]] | None = None,
    labels_path: str | None = None,
    enrollment_path: str | None = None,
    age_bins: tuple[int, ...] = DEFAULT_AGE_BINS,
    n_boot: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
    within_user_aggregation: str = "micro",
) -> dict[str, pd.DataFrame]:
    """Paired user-bootstrap CIs + point estimate for the fair skill score.

    Args:
        models: ``{name: {"path": metrics_dir, "display_name": ...}}``.
        baseline_model: key in ``models`` used as the disparity-ratio denominator.
        continuous_metrics: metric keys scored on continuous channels (e.g. mae).
        binary_metrics: metric keys scored on binary channels (e.g. auprc).
        continuous_channel_indices: continuous channels to score.
        binary_channel_indices: binary channels to score.
        attrs: sensitive attributes to score + macro-average (age_group, sex).
        demographics: optional precomputed ``{user_id: {attr: subgroup}}`` map;
            loaded from ``labels_path``/``enrollment_path`` when omitted.
        labels_path: labels JSON for demographics (required if ``demographics`` is None).
        enrollment_path: enrollment JSON for demographics (required if None).
        age_bins: age-group bin edges for demographics.
        n_boot: number of bootstrap draws.
        seed: master RNG seed (a per-run seed is derived deterministically).
        ci_level: percentile-CI level (0.95 -> 2.5/97.5).
        clip_lower: lower clip on the per-task disparity ratio.
        clip_upper: upper clip on the per-task disparity ratio.
        within_user_aggregation: 'micro' (default) weights each window by its finite
            horizon-cell count when building per-user errors; 'macro' averages
            per-window means unweighted (legacy). Shared with the point flow.

    Returns:
        ``{"fairness_skill_scores": ci_df, "fairness_skill_scores_point": point_df}``.
        ``ci_df`` is keyed by ``(model, scope)`` with ``mean, se, ci_lo, ci_hi,
        n_boot``; ``point_df`` carries ``fair_skill_score, n_tasks``. ``scope`` is
        one entry per attribute plus ``"overall"``.
    """
    attrs = tuple(attrs)

    # ---- Phase 0: build the per-user error table ONCE (the only disk IO) ----
    error_df = _build_error_table(
        models=models,
        continuous_metrics=continuous_metrics,
        binary_metrics=binary_metrics,
        continuous_channel_indices=continuous_channel_indices,
        binary_channel_indices=binary_channel_indices,
        within_user_aggregation=within_user_aggregation,
    )
    if error_df.empty:
        logger.warning("Fairness bootstrap: no error rows discovered; returning empty tables.")
        return {
            "fairness_skill_scores": pd.DataFrame(columns=_SUMMARY_COLUMNS),
            "fairness_skill_scores_point": pd.DataFrame(
                columns=["model", "scope", "fair_skill_score", "n_tasks"]
            ),
        }

    if demographics is None:
        if labels_path is None or enrollment_path is None:
            raise ValueError("labels_path and enrollment_path are required without demographics")
        demographics = load_user_demographics(
            user_ids=set(error_df["user_id"].astype(str)),
            labels_path=labels_path,
            enrollment_path=enrollment_path,
            age_bins=tuple(age_bins),
        )

    # Deterministic point estimate (shares error_df with the draws below).
    point_df = compute_fair_skill_scores_from_errors(
        error_df,
        demographics,
        attrs=attrs,
        baseline_method=baseline_model,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )

    # ---- Phase 1: one shared user-resample matrix ----
    users = sorted(set(error_df["user_id"].astype(str)))
    n_users = len(users)
    if n_users == 0:
        return {
            "fairness_skill_scores": pd.DataFrame(columns=_SUMMARY_COLUMNS),
            "fairness_skill_scores_point": point_df,
        }
    idx_b = _bootstrap_indices(n_users, n_boot, _seed_for(seed, "forecasting"))
    logger.info("Forecasting fairness bootstrap: U=%d users, B=%d, seed=%d", n_users, n_boot, seed)

    # ---- Phase 2: per-draw recompute via the deterministic core ----
    records: list[dict] = []
    for b in range(n_boot):
        replicas = _draw_replica_frame(users, idx_b[b])
        # Remap demographics onto the replica ids (<uid>#r0, ...) so the ported
        # core, which maps user_id -> demographics, runs unchanged after resample.
        demo_b = {
            unit: demographics.get(str(orig), {})
            for orig, unit in zip(replicas["user_id"], replicas["_unit"])
        }
        err_b = _resample(error_df, replicas, "user_id")
        long_b = _build_subgroup_error_long(err_b, demo_b, attrs=attrs)
        fair_b = compute_fair_skill_scores(
            long_b,
            attrs=attrs,
            baseline_method=baseline_model,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        for _, row in fair_b.iterrows():
            records.append(
                {
                    "model": row["model"],
                    "scope": row["scope"],
                    "draw": b,
                    "value": float(row["fair_skill_score"]),
                }
            )

    return {
        "fairness_skill_scores": _summary_table(records, ["model", "scope"], ci_level),
        "fairness_skill_scores_point": point_df,
    }
