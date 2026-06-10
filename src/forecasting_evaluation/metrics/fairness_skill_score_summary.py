"""Compute forecasting fairness-adjusted skill scores.

.. deprecated::
    The ``S_overall − λ·D`` "fairness-adjusted skill score" computed here is the
    legacy "Family B" metric. The default fairness metric is now the
    disparity-ratio **Fairness Skill Score** in
    :mod:`forecasting_evaluation.metrics.fair_skill_score` (point) and
    :mod:`forecasting_evaluation.metrics.bootstrap_fair_skill_score` (bootstrap
    CIs). This module is kept callable for back-compat; its demographics helpers
    (``load_user_demographics``, ``bin_age``, ``normalize_sex``) and
    ``_build_error_table`` are reused by the new metric.

This paper-result helper follows the unified scoring definition used for the
benchmark: task errors are converted to ratios against a fixed baseline,
clipped, then aggregated with a geometric mean. For subgroup scores, model
errors are subgroup-specific while baseline errors remain global full-test
errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics.skill_score_summary import (  # noqa: E402
    _channel_label,
    _list_parquet_files,
    _load_models_dict,
    _metric_channel_value,
    _metric_to_error,
    _parse_channel_indices,
    _safe_read_parquet,
    _safe_to_metric_array,
)

DEFAULT_AGE_BINS = (18, 30, 40, 50, 60)
DEFAULT_DEMOGRAPHIC_ATTRS = ("age_group", "sex")

# Enrollment exposes only a birth *year*, so age is anchored to a single
# reference date shared with Track 1 (downstream_evaluation.config.LABEL_REFERENCE_DATE),
# i.e. age = AGE_REFERENCE_DATE.year − birth_year for every user.
AGE_REFERENCE_DATE = "2020-06-01"


def _last_value(entries: dict[str, Any], user_id: str) -> Any | None:
    item = entries.get(user_id)
    if not isinstance(item, dict):
        return None
    values = item.get("values") or []
    if not values:
        return None
    return values[-1]


def bin_age(age: float | int | None, age_bins: tuple[int, ...] = DEFAULT_AGE_BINS) -> str:
    """Bin an age into the benchmark demographic buckets."""
    if age is None or not np.isfinite(float(age)):
        return "unknown"
    age_value = float(age)
    for idx in range(len(age_bins) - 1):
        if age_bins[idx] <= age_value < age_bins[idx + 1]:
            return f"{age_bins[idx]}-{age_bins[idx + 1] - 1}"
    if age_value >= age_bins[-1]:
        return f"{age_bins[-1]}+"
    return "unknown"


def normalize_sex(value: Any) -> str:
    """Normalize BiologicalSex values to male/female/unknown."""
    if isinstance(value, bool):
        return "male" if value else "female"
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        if int(value) == 1:
            return "male"
        if int(value) == 0:
            return "female"
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"male", "m", "man", "true", "1"}:
            return "male"
        if token in {"female", "f", "woman", "false", "0"}:
            return "female"
    return "unknown"


def load_user_demographics(
    *,
    user_ids: set[str],
    labels_path: str | Path,
    enrollment_path: str | Path,
    age_bins: tuple[int, ...] = DEFAULT_AGE_BINS,
    reference_date: str = AGE_REFERENCE_DATE,
) -> dict[str, dict[str, str]]:
    """Load age-group and sex labels for the requested users.

    Age is ``reference_date.year − birth_year`` (enrollment exposes only a birth
    *year*), anchoring every user to the same reference as Track 1. Sex is the
    latest ``BiologicalSex`` label value.
    """
    labels_file = Path(labels_path)
    enrollment_file = Path(enrollment_path)
    with labels_file.open("r", encoding="utf-8") as file:
        labels_data = json.load(file)
    with enrollment_file.open("r", encoding="utf-8") as file:
        enrollment = json.load(file)

    reference_year = pd.Timestamp(reference_date).year
    sex_entries = labels_data.get("BiologicalSex", {})

    demographics: dict[str, dict[str, str]] = {}
    for user_id in sorted(user_ids):
        record = enrollment.get(user_id)
        birth_year = record.get("birth_year") if isinstance(record, dict) else None
        age = reference_year - int(birth_year) if birth_year else None
        sex_value = _last_value(sex_entries, user_id)
        demographics[user_id] = {
            "age_group": bin_age(age, age_bins=age_bins),
            "sex": normalize_sex(sex_value),
        }
    return demographics


def _load_metric_values(
    *,
    model_name: str,
    model_root: str | Path,
    metric_name: str,
    channel_indices: tuple[int, ...],
    group_name: str,
) -> pd.DataFrame:
    metric_dir = Path(model_root) / metric_name
    per_user_values: dict[tuple[str, int, str], list[float]] = {}

    for parquet_file in _list_parquet_files(metric_dir):
        df = _safe_read_parquet(
            parquet_file,
            columns=["user_id", "history_length", "forecasting_length", metric_name],
        )
        if df is None or "user_id" not in df.columns or metric_name not in df.columns:
            continue

        for _, row in df.iterrows():
            user_id = str(row.get("user_id"))
            metric = _safe_to_metric_array(row.get(metric_name))
            if metric is None:
                continue
            for channel_idx in channel_indices:
                value = _metric_channel_value(metric=metric, channel_idx=channel_idx)
                if not np.isfinite(value):
                    continue
                error = _metric_to_error(metric_name=metric_name, metric_value=value)
                if not np.isfinite(error):
                    continue
                key = (user_id, int(channel_idx), metric_name)
                per_user_values.setdefault(key, []).append(error)

    rows: list[dict[str, Any]] = []
    for (user_id, channel_idx, metric), values in per_user_values.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "group": group_name,
                "metric": metric,
                "channel_idx": int(channel_idx),
                "channel_name": _channel_label(channel_idx),
                "user_id": user_id,
                "error": float(np.mean(finite)),
                "n_values": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def _build_error_table(
    *,
    models: dict[str, dict[str, str]],
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_channel_indices: tuple[int, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    metric_groups = {
        "continuous": (
            [metric.strip().lower() for metric in continuous_metrics if metric.strip()],
            continuous_channel_indices,
        ),
        "binary": (
            [metric.strip().lower() for metric in binary_metrics if metric.strip()],
            binary_channel_indices,
        ),
    }
    for group_name, (metric_names, channel_indices) in metric_groups.items():
        for model_name, model_spec in models.items():
            for metric_name in metric_names:
                frame = _load_metric_values(
                    model_name=model_name,
                    model_root=model_spec["path"],
                    metric_name=metric_name,
                    channel_indices=channel_indices,
                    group_name=group_name,
                )
                if not frame.empty:
                    frames.append(frame)

    columns = [
        "model",
        "group",
        "metric",
        "channel_idx",
        "channel_name",
        "user_id",
        "error",
        "n_values",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _task_cols() -> list[str]:
    return ["group", "metric", "channel_idx", "channel_name"]


def _score_from_ratios(ratios: np.ndarray, clip_lower: float, clip_upper: float) -> float:
    valid = np.asarray(ratios, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    if valid.size == 0:
        return float("nan")
    clipped = np.clip(valid, float(clip_lower), float(clip_upper))
    return float(1.0 - np.exp(np.mean(np.log(clipped))))


def _build_task_errors(error_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame(columns=["model", *_task_cols(), "error", "n_users"])
    grouped = error_df.groupby(["model", *_task_cols()], sort=True)
    return grouped.agg(
        error=("error", "mean"),
        n_users=("user_id", "nunique"),
    ).reset_index()


def _compute_global_scores(
    *,
    task_errors: pd.DataFrame,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    columns = [
        "model",
        "skill_score",
        "geometric_mean_ratio",
        "n_tasks",
        "mean_error",
        "baseline_error_mean",
    ]
    if task_errors.empty:
        return pd.DataFrame(columns=columns)

    baseline = task_errors.loc[task_errors["model"] == baseline_model].copy()
    if baseline.empty:
        raise ValueError(f"Baseline model '{baseline_model}' has no readable metric rows.")
    baseline_lookup = baseline.set_index(_task_cols())["error"]

    rows: list[dict[str, Any]] = []
    for model_name in models:
        model_tasks = task_errors.loc[task_errors["model"] == model_name]
        ratios: list[float] = []
        model_errors: list[float] = []
        baseline_errors: list[float] = []
        for _, row in model_tasks.iterrows():
            key = tuple(row[col] for col in _task_cols())
            if key not in baseline_lookup.index:
                continue
            baseline_error = float(baseline_lookup.loc[key])
            model_error = float(row["error"])
            if baseline_error <= 0 or not np.isfinite(baseline_error):
                continue
            if not np.isfinite(model_error):
                continue
            ratios.append(model_error / baseline_error)
            model_errors.append(model_error)
            baseline_errors.append(baseline_error)

        ratios_arr = np.asarray(ratios, dtype=float)
        skill = _score_from_ratios(ratios_arr, clip_lower, clip_upper)
        gm_ratio = (
            float(np.exp(np.mean(np.log(np.clip(ratios_arr, clip_lower, clip_upper)))))
            if ratios_arr.size
            else float("nan")
        )
        rows.append(
            {
                "model": model_name,
                "skill_score": skill,
                "geometric_mean_ratio": gm_ratio,
                "n_tasks": int(ratios_arr.size),
                "mean_error": float(np.mean(model_errors)) if model_errors else float("nan"),
                "baseline_error_mean": (
                    float(np.mean(baseline_errors)) if baseline_errors else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _compute_average_ranks(task_errors: pd.DataFrame) -> pd.DataFrame:
    if task_errors.empty:
        return pd.DataFrame(columns=["model", "avg_rank", "n_ranked_tasks"])
    ranked = task_errors.copy()
    ranked["rank"] = ranked.groupby(_task_cols())["error"].rank(
        method="average",
        ascending=True,
    )
    return ranked.groupby("model", sort=True).agg(
        avg_rank=("rank", "mean"),
        n_ranked_tasks=("rank", "count"),
    ).reset_index()


def _compute_subgroup_scores(
    *,
    error_df: pd.DataFrame,
    demographics: dict[str, dict[str, str]],
    models: dict[str, dict[str, str]],
    baseline_model: str,
    clip_lower: float,
    clip_upper: float,
    demographic_attrs: tuple[str, ...] = DEFAULT_DEMOGRAPHIC_ATTRS,
) -> pd.DataFrame:
    columns = [
        "model",
        "demographic_attr",
        "subgroup",
        "skill_score",
        "geometric_mean_ratio",
        "n_tasks",
        "n_units",
    ]
    if error_df.empty:
        return pd.DataFrame(columns=columns)

    baseline_task_errors = _build_task_errors(error_df.loc[error_df["model"] == baseline_model])
    baseline_lookup = baseline_task_errors.set_index(_task_cols())["error"]

    with_demo = error_df.copy()
    for attr in demographic_attrs:
        with_demo[attr] = with_demo["user_id"].map(
            lambda uid, attr=attr: demographics.get(str(uid), {}).get(attr, "unknown")
        )

    rows: list[dict[str, Any]] = []
    for attr in demographic_attrs:
        for (model_name, subgroup), group_df in with_demo.groupby(["model", attr], sort=True):
            task_means = group_df.groupby(_task_cols(), sort=True).agg(
                error=("error", "mean"),
                n_units=("user_id", "nunique"),
            ).reset_index()
            ratios: list[float] = []
            n_units = set(group_df["user_id"].astype(str).tolist())
            for _, row in task_means.iterrows():
                key = tuple(row[col] for col in _task_cols())
                if key not in baseline_lookup.index:
                    continue
                baseline_error = float(baseline_lookup.loc[key])
                model_error = float(row["error"])
                if baseline_error <= 0 or not np.isfinite(baseline_error):
                    continue
                if not np.isfinite(model_error):
                    continue
                ratios.append(model_error / baseline_error)

            ratios_arr = np.asarray(ratios, dtype=float)
            skill = _score_from_ratios(ratios_arr, clip_lower, clip_upper)
            gm_ratio = (
                float(np.exp(np.mean(np.log(np.clip(ratios_arr, clip_lower, clip_upper)))))
                if ratios_arr.size
                else float("nan")
            )
            rows.append(
                {
                    "model": model_name,
                    "demographic_attr": attr,
                    "subgroup": str(subgroup),
                    "skill_score": skill,
                    "geometric_mean_ratio": gm_ratio,
                    "n_tasks": int(ratios_arr.size),
                    "n_units": len(n_units),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    ordered_models = list(models)
    result = pd.DataFrame(rows, columns=columns)
    result["model"] = pd.Categorical(result["model"], categories=ordered_models, ordered=True)
    return result.sort_values(["demographic_attr", "model", "subgroup"]).reset_index(drop=True)


def _build_fairness_summary(
    *,
    subgroup_df: pd.DataFrame,
    global_df: pd.DataFrame,
    lambda_fairness: float,
) -> pd.DataFrame:
    columns = [
        "model",
        "demographic_attr",
        "S_overall",
        "disparity",
        "best_group",
        "worst_group",
        "lambda",
        "attr_fairness_adjusted_score",
    ]
    rows: list[dict[str, Any]] = []
    global_lookup = global_df.set_index("model")["skill_score"] if not global_df.empty else {}

    for (model_name, attr), group_df in subgroup_df.groupby(
        ["model", "demographic_attr"],
        sort=True,
        observed=False,
    ):
        valid = group_df[np.isfinite(group_df["skill_score"])]
        if valid.shape[0] >= 2:
            best_idx = valid["skill_score"].idxmax()
            worst_idx = valid["skill_score"].idxmin()
            best_group = str(valid.loc[best_idx, "subgroup"])
            worst_group = str(valid.loc[worst_idx, "subgroup"])
            disparity = float(valid.loc[best_idx, "skill_score"] - valid.loc[worst_idx, "skill_score"])
        else:
            best_group = ""
            worst_group = ""
            disparity = float("nan")

        overall = float(global_lookup.loc[model_name]) if model_name in global_lookup.index else float("nan")
        rows.append(
            {
                "model": str(model_name),
                "demographic_attr": attr,
                "S_overall": overall,
                "disparity": disparity,
                "best_group": best_group,
                "worst_group": worst_group,
                "lambda": float(lambda_fairness),
                "attr_fairness_adjusted_score": overall - float(lambda_fairness) * disparity,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _build_model_summary(
    *,
    global_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    ranks_df: pd.DataFrame,
    models: dict[str, dict[str, str]],
    lambda_fairness: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_lookup = global_df.set_index("model") if not global_df.empty else pd.DataFrame()
    ranks_lookup = ranks_df.set_index("model") if not ranks_df.empty else pd.DataFrame()

    for model_name in models:
        overall = (
            float(global_lookup.loc[model_name, "skill_score"])
            if model_name in global_lookup.index
            else float("nan")
        )
        avg_rank = (
            float(ranks_lookup.loc[model_name, "avg_rank"])
            if model_name in ranks_lookup.index
            else float("nan")
        )
        n_ranked_tasks = (
            int(ranks_lookup.loc[model_name, "n_ranked_tasks"])
            if model_name in ranks_lookup.index
            else 0
        )
        model_fairness = fairness_df.loc[fairness_df["model"] == model_name]
        disparities = {
            str(row["demographic_attr"]): float(row["disparity"])
            for _, row in model_fairness.iterrows()
            if np.isfinite(float(row["disparity"]))
        }
        mean_disparity = (
            float(np.mean(list(disparities.values()))) if disparities else float("nan")
        )
        rows.append(
            {
                "model": model_name,
                "S_overall": overall,
                "avg_rank": avg_rank,
                "n_ranked_tasks": n_ranked_tasks,
                "age_group_disparity": disparities.get("age_group", float("nan")),
                "sex_disparity": disparities.get("sex", float("nan")),
                "mean_disparity": mean_disparity,
                "lambda": float(lambda_fairness),
                "fairness_adjusted_skill_score": overall
                - float(lambda_fairness) * mean_disparity,
            }
        )
    return pd.DataFrame(rows)


def _build_channel_summary(
    *,
    error_df: pd.DataFrame,
    task_errors: pd.DataFrame,
    demographics: dict[str, dict[str, str]],
    models: dict[str, dict[str, str]],
    baseline_model: str,
    clip_lower: float,
    clip_upper: float,
    lambda_fairness: float,
) -> pd.DataFrame:
    columns = [
        "model",
        "group",
        "channel_idx",
        "channel_name",
        "S_overall",
        "avg_rank",
        "n_ranked_tasks",
        "age_group_disparity",
        "sex_disparity",
        "mean_disparity",
        "lambda",
        "fairness_adjusted_skill_score",
    ]
    if error_df.empty or task_errors.empty:
        return pd.DataFrame(columns=columns)

    channel_keys = (
        task_errors[["group", "channel_idx", "channel_name"]]
        .drop_duplicates()
        .sort_values(["group", "channel_idx"])
        .to_dict("records")
    )
    frames: list[pd.DataFrame] = []
    for channel in channel_keys:
        group_name = channel["group"]
        channel_idx = int(channel["channel_idx"])
        channel_name = channel["channel_name"]
        task_slice = task_errors.loc[
            (task_errors["group"] == group_name)
            & (task_errors["channel_idx"] == channel_idx)
        ]
        error_slice = error_df.loc[
            (error_df["group"] == group_name)
            & (error_df["channel_idx"] == channel_idx)
        ]
        channel_global_df = _compute_global_scores(
            task_errors=task_slice,
            models=models,
            baseline_model=baseline_model,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        channel_ranks_df = _compute_average_ranks(task_slice)
        channel_subgroup_df = _compute_subgroup_scores(
            error_df=error_slice,
            demographics=demographics,
            models=models,
            baseline_model=baseline_model,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        channel_fairness_df = _build_fairness_summary(
            subgroup_df=channel_subgroup_df,
            global_df=channel_global_df,
            lambda_fairness=lambda_fairness,
        )
        channel_model_df = _build_model_summary(
            global_df=channel_global_df,
            fairness_df=channel_fairness_df,
            ranks_df=channel_ranks_df,
            models=models,
            lambda_fairness=lambda_fairness,
        )
        channel_model_df.insert(1, "group", group_name)
        channel_model_df.insert(2, "channel_idx", channel_idx)
        channel_model_df.insert(3, "channel_name", channel_name)
        frames.append(channel_model_df)

    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    return result[columns].sort_values(["group", "channel_idx", "model"]).reset_index(drop=True)


def compute_fairness_skill_score_tables(
    *,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_channel_indices: tuple[int, ...],
    clip_lower: float,
    clip_upper: float,
    lambda_fairness: float,
    labels_path: str | Path | None = None,
    enrollment_path: str | Path | None = None,
    demographics: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute subgroup, fairness, model-summary, and channel-summary tables."""
    if baseline_model not in models:
        raise ValueError(
            f"Baseline model '{baseline_model}' is not in model config. "
            f"Available models: {', '.join(models)}"
        )
    if clip_lower <= 0 or clip_upper <= 0 or clip_lower > clip_upper:
        raise ValueError("clip bounds must be positive with lower <= upper")
    if lambda_fairness < 0:
        raise ValueError("lambda_fairness must be non-negative")

    error_df = _build_error_table(
        models=models,
        continuous_metrics=continuous_metrics,
        binary_metrics=binary_metrics,
        continuous_channel_indices=continuous_channel_indices,
        binary_channel_indices=binary_channel_indices,
    )
    task_errors = _build_task_errors(error_df)
    global_df = _compute_global_scores(
        task_errors=task_errors,
        models=models,
        baseline_model=baseline_model,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    ranks_df = _compute_average_ranks(task_errors)

    if demographics is None:
        if labels_path is None or enrollment_path is None:
            raise ValueError("labels_path and enrollment_path are required without demographics")
        demographics = load_user_demographics(
            user_ids=set(error_df["user_id"].astype(str).tolist()),
            labels_path=labels_path,
            enrollment_path=enrollment_path,
        )

    subgroup_df = _compute_subgroup_scores(
        error_df=error_df,
        demographics=demographics,
        models=models,
        baseline_model=baseline_model,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    fairness_df = _build_fairness_summary(
        subgroup_df=subgroup_df,
        global_df=global_df,
        lambda_fairness=lambda_fairness,
    )
    model_summary_df = _build_model_summary(
        global_df=global_df,
        fairness_df=fairness_df,
        ranks_df=ranks_df,
        models=models,
        lambda_fairness=lambda_fairness,
    )
    channel_summary_df = _build_channel_summary(
        error_df=error_df,
        task_errors=task_errors,
        demographics=demographics,
        models=models,
        baseline_model=baseline_model,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        lambda_fairness=lambda_fairness,
    )
    return subgroup_df, fairness_df, model_summary_df, channel_summary_df


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute forecasting fairness-adjusted skill scores."
    )
    parser.add_argument("--config", default=None, help="JSON/YAML config with model mappings.")
    parser.add_argument("--models-json", default=None, help="Inline JSON model mapping.")
    parser.add_argument("--baseline", required=True, help="Baseline model key from config.")
    parser.add_argument("--continuous-channel-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--continuous-metrics", nargs="+", default=["mase", "sql"])
    parser.add_argument("--binary-channel-indices", default="7,8,9,10,11,12,13,14,15,16,17,18")
    parser.add_argument("--binary-metrics", nargs="+", default=["mse", "f1"])
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=100.0)
    parser.add_argument("--lambda-fairness", type=float, default=0.5)
    parser.add_argument("--labels-path", default="data/labels/last_labels.json")
    parser.add_argument("--enrollment-path", default="data/labels/enrollment_info.json")
    parser.add_argument("--output-dir", default="results/metrics_summary")
    parser.add_argument("--output-prefix", default="paper_forecasting_fairness_skill_score")
    return parser


def main() -> None:
    """Generate forecasting fairness skill score CSV outputs."""
    args = build_parser().parse_args()
    models = _load_models_dict(args)
    continuous_channel_indices = _parse_channel_indices(
        args.continuous_channel_indices,
        default=tuple(range(0, 7)),
    )
    binary_channel_indices = _parse_channel_indices(
        args.binary_channel_indices,
        default=tuple(range(7, 19)),
    )

    subgroup_df, fairness_df, model_summary_df, channel_summary_df = (
        compute_fairness_skill_score_tables(
            models=models,
            baseline_model=args.baseline,
            continuous_metrics=list(args.continuous_metrics),
            binary_metrics=list(args.binary_metrics),
            continuous_channel_indices=continuous_channel_indices,
            binary_channel_indices=binary_channel_indices,
            clip_lower=float(args.clip_lower),
            clip_upper=float(args.clip_upper),
            lambda_fairness=float(args.lambda_fairness),
            labels_path=args.labels_path,
            enrollment_path=args.enrollment_path,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subgroup_path = output_dir / f"{args.output_prefix}_subgroup_scores.csv"
    fairness_path = output_dir / f"{args.output_prefix}_fairness_summary.csv"
    model_summary_path = output_dir / f"{args.output_prefix}_model_summary.csv"
    channel_summary_path = output_dir / f"{args.output_prefix}_channel_summary.csv"

    subgroup_df.to_csv(subgroup_path, index=False)
    fairness_df.to_csv(fairness_path, index=False)
    model_summary_df.to_csv(model_summary_path, index=False)
    channel_summary_df.to_csv(channel_summary_path, index=False)

    print("=== Forecasting fairness skill score summary ===")
    if model_summary_df.empty:
        print("(empty)")
    else:
        print(model_summary_df.to_string(index=False))
    print(f"\nSaved subgroup scores: {subgroup_path}")
    print(f"Saved fairness summary: {fairness_path}")
    print(f"Saved model summary: {model_summary_path}")
    print(f"Saved channel summary: {channel_summary_path}")
    print(f"Baseline: {args.baseline}")
    print(f"Ratio clip: [{args.clip_lower}, {args.clip_upper}]")
    print(f"Lambda fairness: {args.lambda_fairness}")


if __name__ == "__main__":
    main()
