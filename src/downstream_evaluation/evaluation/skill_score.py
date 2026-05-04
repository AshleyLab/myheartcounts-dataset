"""Skill Score: unified evaluation metric across heterogeneous tasks.

Computes relative improvement over a baseline using the geometric mean of
clipped error ratios, following FEVBench (Shchur et al., 2025).

    S_j = 1 - GeoMean( clip(E_rj / E_rβ, ℓ, u) )

where E_rj is the error of model j on task r, E_rβ is the baseline error,
and clip bounds ratios to [ℓ, u] (default [0.01, 100]).

For higher-is-better metrics (AUROC, AUPRC, Pearson R, Spearman R),
errors are pseudo-errors: E = 1 - Metric.  For lower-is-better metrics
(MAE, MSE), E is the metric directly.

A skill score of 0 means on par with the baseline, 0.2 means a 20% average
error reduction, and negative means worse than the baseline.

Aggregation convention: the ``overall`` skill score (and overall average rank)
is **domain-balanced** — the arithmetic mean of per-domain skill scores —
so that domains with more tasks (e.g. Medical conditions, 12/33) do not
dominate the headline number. A task-flat (micro) aggregate is intentionally
not surfaced on the public dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task → domain mapping (41 tasks, 6 domains)
# ---------------------------------------------------------------------------

TASK_DOMAIN_MAP: dict[str, str] = {
    # Demographics (2)
    "age": "Demographics",
    "BiologicalSex": "Demographics",
    # Medical conditions (12)
    "cardiovascular_disease": "Medical conditions",
    "Heart Failure or CHF": "Medical conditions",
    "Atrial fibrillation (Afib)": "Medical conditions",
    "PH": "Medical conditions",
    "CAD": "Medical conditions",
    "Congenital Heart": "Medical conditions",
    "Peripheral/Systemic Vascular Disease": "Medical conditions",
    "Cerebrovascular Disease": "Medical conditions",
    "Diabetes": "Medical conditions",
    "Hypertension": "Medical conditions",
    "sleep_diagnosis1": "Medical conditions",
    "framingham_risk": "Medical conditions",
    # Body metrics and biomarkers (8)
    "blood_pressure_categories": "Body metrics and biomarkers",
    "Hdl": "Body metrics and biomarkers",
    "Ldl": "Body metrics and biomarkers",
    "TotalCholesterol": "Body metrics and biomarkers",
    "WeightKilograms": "Body metrics and biomarkers",
    "BMI_categories": "Body metrics and biomarkers",
    "BMI_values": "Body metrics and biomarkers",
    "SystolicBloodPressure": "Body metrics and biomarkers",
    # Mental well-being (7)
    "feel_worthwhile1": "Mental well-being",
    "feel_worthwhile2": "Mental well-being",
    "feel_worthwhile3": "Mental well-being",
    "feel_worthwhile4": "Mental well-being",
    "happiness_categories": "Mental well-being",
    "happiness": "Mental well-being",
    "satisfiedwith_life": "Mental well-being",
    # Wearable physiology (7)
    "Watch_RestingHeartRate": "Wearable physiology",
    "Watch_VO2Max": "Wearable physiology",
    "Watch_HeartRateVariabilitySDNN": "Wearable physiology",
    "Watch_WalkingHeartRateAverage": "Wearable physiology",
    "Watch_StandTime": "Wearable physiology",
    "Watch_BasalEnergyBurned": "Wearable physiology",
    "Watch_RespiratoryRate": "Wearable physiology",
    # Sleep and lifestyle (5)
    "WakeUpTime_categories": "Sleep and lifestyle",
    "GoSleepTime_categories": "Sleep and lifestyle",
    "sleep_time_categories": "Sleep and lifestyle",
    "work": "Sleep and lifestyle",
    "vigorous_act": "Sleep and lifestyle",
}

# Default primary metric per task type (all higher-is-better for downstream).
# For downstream tasks: E = 1 - Metric.
DEFAULT_PRIMARY_METRICS: dict[str, str] = {
    "binary": "auprc",
    "ordinal": "spearman_r",
    "regression": "pearson_r",
}

# Default clipping bounds for error ratios.
DEFAULT_CLIP_LOWER = 1e-2
DEFAULT_CLIP_UPPER = 100.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskError:
    """Error and ratio for a single task."""

    task: str
    task_type: str
    domain: str
    metric_name: str
    model_metric: float
    baseline_metric: float
    model_error: float
    baseline_error: float
    ratio: float  # clipped E_model / E_baseline
    higher_is_better: bool


@dataclass
class SkillScoreResult:
    """Complete skill score result with per-domain and per-task breakdown.

    ``overall`` is the **domain-balanced** (macro) aggregate: the arithmetic
    mean of per-domain skill scores. Each domain counts equally regardless
    of task count. A task-flat (micro) aggregate is intentionally not
    provided; see module docstring.
    """

    overall: float
    domain_scores: dict[str, float]
    n_tasks_per_domain: dict[str, int]
    task_errors: list[TaskError]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_skill_score(
    ratios: np.ndarray,
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
) -> float:
    """Compute skill score from an array of error ratios.

    S = 1 - GeoMean(clip(ratios, ℓ, u))

    Args:
        ratios: Array of E_model / E_baseline for each task.
        clip_lower: Lower bound for clipping.
        clip_upper: Upper bound for clipping.

    Returns:
        Skill score. 0 = on par with baseline, positive = better.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    valid = ratios[np.isfinite(ratios)]
    if len(valid) == 0:
        return float("nan")
    clipped = np.clip(valid, clip_lower, clip_upper)
    geo_mean = np.exp(np.mean(np.log(clipped)))
    return float(1.0 - geo_mean)


def metric_to_error(value: float, higher_is_better: bool = True) -> float:
    """Convert a metric value to an error value.

    For higher-is-better metrics (AUROC, Pearson R, etc.): E = 1 - Metric.
    For lower-is-better metrics (MAE, MSE): E = Metric.
    """
    if higher_is_better:
        return 1.0 - value
    return value


# ---------------------------------------------------------------------------
# Extract best metric per task from results CSV
# ---------------------------------------------------------------------------


def _extract_best_per_task(
    results_df: pd.DataFrame,
    primary_metrics: dict[str, str] | None = None,
    metric_prefix: str = "test",
) -> dict[str, tuple[str, str, float]]:
    """Extract the best primary metric value per task.

    When multiple classifiers exist for one task, picks the one with the
    best primary metric (highest for higher-is-better).

    Args:
        results_df: DataFrame with task, task_type, and metric columns.
        primary_metrics: {task_type: metric_suffix}. Defaults to
            DEFAULT_PRIMARY_METRICS.
        metric_prefix: "val" or "test".

    Returns:
        {task_name: (task_type, metric_col, best_value)}
    """
    if primary_metrics is None:
        primary_metrics = DEFAULT_PRIMARY_METRICS

    result: dict[str, tuple[str, str, float]] = {}

    for task_name in results_df["task"].unique():
        task_rows = results_df[results_df["task"] == task_name]
        task_type = task_rows["task_type"].iloc[0]

        metric_suffix = primary_metrics.get(task_type)
        if metric_suffix is None:
            continue

        metric_col = f"{metric_prefix}_{metric_suffix}"
        if metric_col not in task_rows.columns:
            continue

        # Pick best (highest value — all default primary metrics are
        # higher-is-better).
        best_val = float("-inf")
        for _, row in task_rows.iterrows():
            val = row[metric_col]
            if pd.notna(val) and float(val) > best_val:
                best_val = float(val)

        if np.isfinite(best_val):
            result[task_name] = (task_type, metric_col, best_val)

    return result


# ---------------------------------------------------------------------------
# Downstream skill score
# ---------------------------------------------------------------------------


def compute_downstream_skill_scores(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    domain_map: dict[str, str] | None = None,
    primary_metrics: dict[str, str] | None = None,
    metric_prefix: str = "test",
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
) -> SkillScoreResult:
    """Compute domain-level and overall skill scores for downstream eval.

    Args:
        model_df: Results CSV DataFrame for the model being evaluated.
        baseline_df: Results CSV DataFrame for the baseline model.
        domain_map: {task_name: domain_name}. Defaults to TASK_DOMAIN_MAP.
        primary_metrics: {task_type: metric_suffix}. Defaults to
            DEFAULT_PRIMARY_METRICS.
        metric_prefix: "val" or "test".
        clip_lower: Lower clipping bound for error ratios.
        clip_upper: Upper clipping bound for error ratios.

    Returns:
        SkillScoreResult with overall, per-domain, and per-task breakdown.
    """
    if domain_map is None:
        domain_map = TASK_DOMAIN_MAP
    if primary_metrics is None:
        primary_metrics = DEFAULT_PRIMARY_METRICS

    model_best = _extract_best_per_task(model_df, primary_metrics, metric_prefix)
    baseline_best = _extract_best_per_task(baseline_df, primary_metrics, metric_prefix)

    task_errors: list[TaskError] = []

    for task_name, (task_type, metric_col, model_val) in model_best.items():
        if task_name not in domain_map:
            continue
        if task_name not in baseline_best:
            logger.warning("Task '%s' missing from baseline, skipping", task_name)
            continue

        _, _, baseline_val = baseline_best[task_name]
        domain = domain_map[task_name]

        # All default primary metrics are higher-is-better.
        hib = True
        model_err = metric_to_error(model_val, hib)
        baseline_err = metric_to_error(baseline_val, hib)

        # Avoid division by zero: if baseline is perfect, ratio is undefined.
        if baseline_err <= 0:
            logger.warning(
                "Task '%s': baseline error <= 0 (metric=%.4f), skipping",
                task_name, baseline_val,
            )
            continue

        ratio = np.clip(model_err / baseline_err, clip_lower, clip_upper)

        task_errors.append(TaskError(
            task=task_name,
            task_type=task_type,
            domain=domain,
            metric_name=metric_col,
            model_metric=model_val,
            baseline_metric=baseline_val,
            model_error=model_err,
            baseline_error=baseline_err,
            ratio=float(ratio),
            higher_is_better=hib,
        ))

    # Per-domain skill scores
    domains = sorted({te.domain for te in task_errors})
    domain_scores: dict[str, float] = {}
    n_tasks_per_domain: dict[str, int] = {}

    for domain in domains:
        domain_ratios = np.array([
            te.ratio for te in task_errors if te.domain == domain
        ])
        domain_scores[domain] = compute_skill_score(
            domain_ratios, clip_lower, clip_upper,
        )
        n_tasks_per_domain[domain] = len(domain_ratios)

    # Overall: domain-balanced (macro) — arithmetic mean of per-domain skill
    # scores. Each domain counts equally regardless of task count, so domains
    # with more tasks (e.g. Medical conditions 12/33) do not dominate.
    if domain_scores:
        overall = float(np.mean(list(domain_scores.values())))
    else:
        overall = float("nan")

    return SkillScoreResult(
        overall=overall,
        domain_scores=domain_scores,
        n_tasks_per_domain=n_tasks_per_domain,
        task_errors=task_errors,
    )


# ---------------------------------------------------------------------------
# Fairness-adjusted skill score
# ---------------------------------------------------------------------------


def compute_fairness_adjusted_skill_scores(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    subgroup_col: str,
    domain_map: dict[str, str] | None = None,
    primary_metrics: dict[str, str] | None = None,
    metric_prefix: str = "test",
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
) -> dict[str, SkillScoreResult]:
    """Compute skill scores per demographic subgroup.

    The numerator E_rj^(g) is computed on subgroup-specific data, but the
    denominator E_rβ is always the GLOBAL baseline error (full test set).
    This keeps all subgroup scores on the same scale.

    Args:
        model_df: Per-task results for each subgroup. Must contain a column
            named ``subgroup_col`` identifying the subgroup (e.g. "age_group",
            "sex"). Rows without this column value are treated as the global
            result.
        baseline_df: GLOBAL baseline results (full test set, no subgroup
            column needed). The baseline error is always global.
        subgroup_col: Column name in model_df identifying the subgroup.
        domain_map: {task_name: domain_name}. Defaults to TASK_DOMAIN_MAP.
        primary_metrics: {task_type: metric_suffix}.
        metric_prefix: "val" or "test".
        clip_lower: Lower clipping bound for error ratios.
        clip_upper: Upper clipping bound for error ratios.

    Returns:
        {subgroup_value: SkillScoreResult} for each unique subgroup.
    """
    if domain_map is None:
        domain_map = TASK_DOMAIN_MAP
    if primary_metrics is None:
        primary_metrics = DEFAULT_PRIMARY_METRICS

    if subgroup_col not in model_df.columns:
        raise ValueError(
            f"Column '{subgroup_col}' not found in model_df. "
            f"Available columns: {list(model_df.columns)}"
        )

    # Global baseline errors (shared denominator for all subgroups).
    baseline_best = _extract_best_per_task(baseline_df, primary_metrics, metric_prefix)
    global_baseline_errors: dict[str, float] = {}
    for task_name, (_, _, baseline_val) in baseline_best.items():
        global_baseline_errors[task_name] = metric_to_error(baseline_val, True)

    results: dict[str, SkillScoreResult] = {}

    for subgroup_val, sub_df in model_df.groupby(subgroup_col):
        sub_best = _extract_best_per_task(sub_df, primary_metrics, metric_prefix)
        task_errors: list[TaskError] = []

        for task_name, (task_type, metric_col, model_val) in sub_best.items():
            if task_name not in domain_map:
                continue
            baseline_err = global_baseline_errors.get(task_name)
            if baseline_err is None or baseline_err <= 0:
                continue

            domain = domain_map[task_name]
            model_err = metric_to_error(model_val, True)
            ratio = np.clip(model_err / baseline_err, clip_lower, clip_upper)

            baseline_val = baseline_best[task_name][2]
            task_errors.append(TaskError(
                task=task_name,
                task_type=task_type,
                domain=domain,
                metric_name=metric_col,
                model_metric=model_val,
                baseline_metric=baseline_val,
                model_error=model_err,
                baseline_error=baseline_err,
                ratio=float(ratio),
                higher_is_better=True,
            ))

        # Aggregate
        domains = sorted({te.domain for te in task_errors})
        domain_scores: dict[str, float] = {}
        n_tasks_per_domain: dict[str, int] = {}
        for domain in domains:
            domain_ratios = np.array([
                te.ratio for te in task_errors if te.domain == domain
            ])
            domain_scores[domain] = compute_skill_score(
                domain_ratios, clip_lower, clip_upper,
            )
            n_tasks_per_domain[domain] = len(domain_ratios)

        # Overall: domain-balanced (macro). See compute_downstream_skill_scores.
        overall = (
            float(np.mean(list(domain_scores.values()))) if domain_scores
            else float("nan")
        )

        results[str(subgroup_val)] = SkillScoreResult(
            overall=overall,
            domain_scores=domain_scores,
            n_tasks_per_domain=n_tasks_per_domain,
            task_errors=task_errors,
        )

    return results
