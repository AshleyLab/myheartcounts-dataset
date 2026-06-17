"""Demographic infrastructure for the forecasting fairness metric.

This module hosts the helpers shared by
:mod:`forecasting_evaluation.metrics.fair_skill_score` and
:mod:`forecasting_evaluation.metrics.bootstrap_fair_skill_score`:

* per-user-error table builder (``_build_error_table``), which loads the
  per-model metric parquet trees produced by ``mhc-forecast-eval`` and yields
  one row per ``(model, task, user)``;
* demographic loading + binning helpers (``load_user_demographics``,
  ``bin_age``, ``normalize_sex``), which map user ids to ``age_group`` and
  ``sex`` subgroup values; and
* task-key utilities (``_task_cols``, ``DEFAULT_AGE_BINS``,
  ``AGE_REFERENCE_DATE``).

The fairness scoring math itself (worst-group skill, macro-mean across attrs)
lives in :mod:`fair_skill_score`; the bootstrap CIs in
:mod:`bootstrap_fair_skill_score`. The legacy ``S − λ·D`` "fairness-adjusted
skill score" that used to live here has been removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics.skill_score_summary import (
    _aggregate_unit_error,
    _channel_label,
    _list_parquet_files,
    _metric_channel_sum_count,
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
    within_user_aggregation: str = "micro",
) -> pd.DataFrame:
    metric_dir = Path(model_root) / metric_name
    # Per (user, channel, metric): one (cell_sum, cell_count) pair per window.
    per_user_pairs: dict[tuple[str, int, str], list[tuple[float, int]]] = {}

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
                sum_count = _metric_channel_sum_count(metric=metric, channel_idx=channel_idx)
                if sum_count is None:
                    continue
                key = (user_id, int(channel_idx), metric_name)
                per_user_pairs.setdefault(key, []).append(sum_count)

    rows: list[dict[str, Any]] = []
    for (user_id, channel_idx, metric), pairs in per_user_pairs.items():
        error, n_values = _aggregate_unit_error(metric, pairs, within_user_aggregation)
        if not np.isfinite(error):
            continue
        rows.append(
            {
                "model": model_name,
                "group": group_name,
                "metric": metric,
                "channel_idx": int(channel_idx),
                "channel_name": _channel_label(channel_idx),
                "user_id": user_id,
                "error": error,
                "n_values": int(n_values),
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
    within_user_aggregation: str = "micro",
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
                    within_user_aggregation=within_user_aggregation,
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
    """Column names that identify a single forecasting task in the error frame."""
    return ["group", "metric", "channel_idx", "channel_name"]
