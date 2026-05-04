"""GlobalScore: type-balanced skill score for HPO model selection.

Normalises each task-level metric into a skill score relative to a null
predictor, then aggregates with type-balanced averaging so that binary,
ordinal, and regression task types contribute equally.

Definitions
-----------
Binary tasks:
    s_bin_t = (AUPRC_t - p_t) / (1 - p_t)
    where p_t is the train prevalence.  A random classifier scores 0,
    a perfect classifier scores 1, regardless of class imbalance.

Ordinal and regression tasks:
    s_mae_t = 1 - MAE_t(model) / MAE_t(null)
    The null predictor is the train median class (ordinal) or train
    median (regression).  Score of 0 = no improvement over null,
    1 = perfect prediction.

Type-balanced aggregation:
    GlobalScore = (1 / |T|) * sum_{type in T} mean_{t in type}(s_t)
    where T is the set of task types present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default proxy task suite (2 per type for balanced evaluation)
DEFAULT_PROXY_TASKS: dict[str, list[str]] = {
    "binary": ["BiologicalSex", "work"],
    "ordinal": ["BMI_categories", "sleep_time_categories"],
    "regression": ["BMI_values", "age"],
}


@dataclass
class TaskSkillScore:
    """Skill score for a single task."""

    task: str
    task_type: str
    raw_metric: float
    null_baseline: float
    skill_score: float


@dataclass
class GlobalScoreResult:
    """Complete GlobalScore computation result with per-task breakdown."""

    global_score: float
    type_scores: dict[str, float]
    task_scores: list[TaskSkillScore]
    n_tasks_per_type: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"GlobalScore: {self.global_score:.4f}"]
        lines.append("  Type scores:")
        for t, s in sorted(self.type_scores.items()):
            lines.append(f"    {t}: {s:.4f} ({self.n_tasks_per_type[t]} tasks)")
        lines.append("  Per-task breakdown:")
        for ts in self.task_scores:
            lines.append(
                f"    {ts.task} ({ts.task_type}): "
                f"skill={ts.skill_score:.4f} "
                f"(metric={ts.raw_metric:.4f}, null={ts.null_baseline:.4f})"
            )
        return "\n".join(lines)


def compute_binary_skill_score(auprc: float, train_prevalence: float) -> float:
    """Normalised AUPRC skill score.

    Args:
        auprc: AUPRC on the evaluation set.
        train_prevalence: Fraction of positive samples in the training set.

    Returns:
        Skill score in [~-p/(1-p), 1]. A random classifier scores ~0.
    """
    if np.isnan(auprc) or np.isnan(train_prevalence):
        return float("nan")
    if train_prevalence >= 1.0:
        return float("nan")
    return (auprc - train_prevalence) / (1.0 - train_prevalence)


def compute_mae_skill_score(model_mae: float, null_mae: float) -> float:
    """MAE-based skill score relative to a null predictor.

    Args:
        model_mae: MAE of the model on the evaluation set.
        null_mae: MAE of the null predictor (train median) on the evaluation set.

    Returns:
        Skill score. 0 = same as null, 1 = perfect, negative = worse than null.
    """
    if np.isnan(model_mae) or np.isnan(null_mae):
        return float("nan")
    if null_mae == 0:
        # Perfect null predictor (all labels identical) — can't improve
        return 0.0
    return 1.0 - model_mae / null_mae


def compute_global_score_from_df(
    results_df: pd.DataFrame,
    train_prevalences: dict[str, float],
    null_maes: dict[str, float],
    proxy_tasks: dict[str, list[str]] | None = None,
    metric_prefix: str = "val",
) -> GlobalScoreResult:
    """Compute GlobalScore from a results DataFrame.

    This is the main entry point for computing GlobalScore from CSV results
    produced by ``run_downstream_eval.py``.

    Args:
        results_df: DataFrame with columns matching ``run_downstream_eval.py`` CSV schema.
            Must contain rows for the proxy tasks.  When multiple classifiers
            exist for a task, the best one (by skill score) is selected.
        train_prevalences: ``{task_name: positive_fraction}`` for binary tasks.
            Computed from training labels.
        null_maes: ``{task_name: null_predictor_mae}`` for ordinal and
            regression tasks.  The null predictor is the train median
            evaluated on the target split (val or test).
        proxy_tasks: ``{task_type: [task_name, ...]}`` defining the proxy
            suite.  Defaults to ``DEFAULT_PROXY_TASKS``.
        metric_prefix: ``"val"`` or ``"test"`` — which split's metrics to use.

    Returns:
        GlobalScoreResult with overall score, per-type scores, and per-task
        breakdown.
    """
    if proxy_tasks is None:
        proxy_tasks = DEFAULT_PROXY_TASKS

    task_scores: list[TaskSkillScore] = []

    for task_type, tasks in proxy_tasks.items():
        for task_name in tasks:
            # Find rows for this task
            task_rows = results_df[results_df["task"] == task_name]
            if task_rows.empty:
                logger.warning("Task '%s' not found in results, skipping", task_name)
                continue

            if task_type == "binary":
                auprc_col = f"{metric_prefix}_auprc"
                if auprc_col not in task_rows.columns:
                    logger.warning("Column '%s' not found, skipping %s", auprc_col, task_name)
                    continue

                prevalence = train_prevalences.get(task_name)
                if prevalence is None:
                    logger.warning("No train prevalence for '%s', skipping", task_name)
                    continue

                # Best classifier for this task (by skill score)
                best_skill = float("-inf")
                best_auprc = float("nan")
                for _, row in task_rows.iterrows():
                    auprc = row[auprc_col]
                    if pd.isna(auprc):
                        continue
                    skill = compute_binary_skill_score(float(auprc), prevalence)
                    if not np.isnan(skill) and skill > best_skill:
                        best_skill = skill
                        best_auprc = float(auprc)

                if np.isinf(best_skill):
                    best_skill = float("nan")

                task_scores.append(
                    TaskSkillScore(
                        task=task_name,
                        task_type=task_type,
                        raw_metric=best_auprc,
                        null_baseline=prevalence,
                        skill_score=best_skill,
                    )
                )

            elif task_type in ("ordinal", "regression"):
                mae_col = (
                    f"{metric_prefix}_mae_ordinal"
                    if task_type == "ordinal"
                    else f"{metric_prefix}_mae"
                )
                if mae_col not in task_rows.columns:
                    logger.warning("Column '%s' not found, skipping %s", mae_col, task_name)
                    continue

                null_mae = null_maes.get(task_name)
                if null_mae is None:
                    logger.warning("No null MAE for '%s', skipping", task_name)
                    continue

                # Best classifier for this task (by skill score, lower MAE = better)
                best_skill = float("-inf")
                best_mae = float("nan")
                for _, row in task_rows.iterrows():
                    mae = row[mae_col]
                    if pd.isna(mae):
                        continue
                    skill = compute_mae_skill_score(float(mae), null_mae)
                    if not np.isnan(skill) and skill > best_skill:
                        best_skill = skill
                        best_mae = float(mae)

                if np.isinf(best_skill):
                    best_skill = float("nan")

                task_scores.append(
                    TaskSkillScore(
                        task=task_name,
                        task_type=task_type,
                        raw_metric=best_mae,
                        null_baseline=null_mae,
                        skill_score=best_skill,
                    )
                )

    # Type-balanced aggregation
    type_scores: dict[str, float] = {}
    n_tasks_per_type: dict[str, int] = {}
    for task_type in proxy_tasks:
        type_task_scores = [
            ts.skill_score
            for ts in task_scores
            if ts.task_type == task_type and not np.isnan(ts.skill_score)
        ]
        if type_task_scores:
            type_scores[task_type] = float(np.mean(type_task_scores))
            n_tasks_per_type[task_type] = len(type_task_scores)
        else:
            type_scores[task_type] = float("nan")
            n_tasks_per_type[task_type] = 0

    # Global score = mean across types (only non-NaN types)
    valid_type_scores = [s for s in type_scores.values() if not np.isnan(s)]
    if valid_type_scores:
        global_score = float(np.mean(valid_type_scores))
    else:
        global_score = float("nan")

    return GlobalScoreResult(
        global_score=global_score,
        type_scores=type_scores,
        task_scores=task_scores,
        n_tasks_per_type=n_tasks_per_type,
    )


def compute_null_baselines(
    labels_df: pd.DataFrame,
    split_users: dict[str, list[str]],
    proxy_tasks: dict[str, list[str]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute null predictor baselines from training data.

    For binary tasks: train prevalence (fraction of positive class).
    For ordinal/regression tasks: MAE of train-median predictor on val set.

    Args:
        labels_df: Labels lookup DataFrame with columns for each task,
            plus ``user_id`` column.
        split_users: ``{"train": [...], "val": [...], "test": [...]}``
            mapping split names to user ID lists.
        proxy_tasks: Task suite definition.  Defaults to ``DEFAULT_PROXY_TASKS``.

    Returns:
        Tuple of ``(train_prevalences, null_maes)`` dicts.
    """
    if proxy_tasks is None:
        proxy_tasks = DEFAULT_PROXY_TASKS

    train_ids = set(split_users["train"])
    # Support both "val" and "validation" key names
    val_key = "val" if "val" in split_users else "validation"
    val_ids = set(split_users[val_key])

    train_prevalences: dict[str, float] = {}
    null_maes: dict[str, float] = {}

    for task_type, tasks in proxy_tasks.items():
        for task_name in tasks:
            if task_name not in labels_df.columns:
                logger.warning("Task '%s' not in labels_df columns, skipping", task_name)
                continue

            # Get train and val labels (drop NaN)
            train_mask = labels_df["user_id"].isin(train_ids)
            val_mask = labels_df["user_id"].isin(val_ids)

            # User-level: take first non-NaN label per user
            train_labels = (
                labels_df.loc[train_mask, ["user_id", task_name]]
                .dropna(subset=[task_name])
                .groupby("user_id")[task_name]
                .first()
            )
            val_labels = (
                labels_df.loc[val_mask, ["user_id", task_name]]
                .dropna(subset=[task_name])
                .groupby("user_id")[task_name]
                .first()
            )

            if train_labels.empty or val_labels.empty:
                logger.warning("No labels for task '%s' in train or val, skipping", task_name)
                continue

            if task_type == "binary":
                train_prevalences[task_name] = float(train_labels.mean())

            elif task_type in ("ordinal", "regression"):
                # Null predictor: train median
                train_median = float(train_labels.median())
                # MAE of predicting train median on val set
                val_values = val_labels.values.astype(float)
                null_mae = float(np.mean(np.abs(val_values - train_median)))
                null_maes[task_name] = null_mae

    return train_prevalences, null_maes
