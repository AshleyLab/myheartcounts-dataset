"""Persist per-(method, task) test predictions + a per-user subgroup map.

When the prediction engine runs with a predictions directory configured, it emits,
per (method, task), a ``test.parquet`` of ``uid, y_true, y_pred, y_proba`` plus a
single shared ``_subgroups.json`` (``{user_id: {age_group, sex}}``). Together these
are the input the paper-metrics bootstrap (``bootstrap_skill_rank``) paired-resamples
for skill / rank / fairness confidence intervals.

Layout::

    <predictions_dir>/<method>/<task>/test.parquet   # task = task name, "/"+" " -> "_"
    <predictions_dir>/_subgroups.json

The module is self-contained: demographics come from the labels lookup's ``age`` /
``BiologicalSex`` columns, so no Labels-API environment is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.evaluation.metrics import get_task_type

# Five age bands (plus ``unknown``) — the subgroups the fairness analysis slices on.
_AGE_GROUP_BINS: tuple[tuple[float, float, str], ...] = (
    (-float("inf"), 30.0, "18-29"),
    (30.0, 40.0, "30-39"),
    (40.0, 50.0, "40-49"),
    (50.0, 60.0, "50-59"),
    (60.0, float("inf"), "60+"),
)

# Sentinels marking a missing label cell in the lookup (mirrors data.provider).
_MISSING_INT = -1
_MISSING_FLOAT = -1.0


def _safe_task(task: str) -> str:
    """Task name as a filesystem-safe directory (matches the bootstrap loader)."""
    return task.replace("/", "_").replace(" ", "_")


def write_task_predictions(
    predictions_dir: str | Path,
    method: str,
    task: str,
    uids,
    y_true,
    y_pred,
) -> None:
    """Write ``<predictions_dir>/<method>/<task>/test.parquet``.

    ``y_pred`` is the evaluator's raw test output: the class-1 probability for
    binary tasks, the point prediction otherwise. Both the discrete ``y_pred`` and
    continuous ``y_proba`` columns are derived from it so the bootstrap can read the
    probability (binary AUPRC) and the point prediction (ordinal Spearman /
    regression Pearson) it needs per task type.
    """
    ttype = get_task_type(task)
    y_true = np.asarray(y_true)
    raw = np.asarray(y_pred, dtype=np.float64)
    if ttype == "binary":
        y_proba = raw
        y_pred_col = (raw >= 0.5).astype(np.int64)
    elif ttype in ("ordinal", "multiclass"):
        y_pred_col = np.round(raw).astype(np.int64)
        y_proba = raw
    else:  # regression
        y_pred_col = raw
        y_proba = raw

    df = (
        pd.DataFrame(
            {
                "uid": np.asarray(uids).astype(str),
                "y_true": y_true,
                "y_pred": y_pred_col,
                "y_proba": y_proba,
            }
        )
        .sort_values("uid")
        .reset_index(drop=True)
    )
    out_dir = Path(predictions_dir) / method / _safe_task(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "test.parquet", index=False)


def _age_group(value) -> str:
    """Bin a numeric age into one of the five bands, else ``unknown``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    for lo, hi, label in _AGE_GROUP_BINS:
        if lo <= v < hi:
            return label
    return "unknown"


def _sex(value) -> str:
    """Map ``BiologicalSex`` (1=male, 0=female) to a subgroup label, else ``unknown``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    return "male" if int(v) == 1 else ("female" if int(v) == 0 else "unknown")


def _first_valid_by_user(lookup: pd.DataFrame, col: str) -> dict[str, float]:
    """``{user_id: first non-sentinel value}`` for a constant-per-user label column."""
    arr = lookup[col].to_numpy()
    if np.issubdtype(arr.dtype, np.floating):
        valid = ~(np.isnan(arr) | (arr == _MISSING_FLOAT))
    else:
        valid = arr != _MISSING_INT
    sub = lookup.loc[valid, ["user_id", col]].drop_duplicates("user_id", keep="first")
    return dict(zip(sub["user_id"].astype(str), sub[col]))


def write_subgroup_map(
    predictions_dir: str | Path,
    lookup_path: str | Path,
    users,
) -> None:
    """Write ``<predictions_dir>/_subgroups.json`` = ``{uid: {age_group, sex}}``.

    Demographics are read from the labels lookup's ``age`` / ``BiologicalSex``
    columns (the first non-sentinel value per user). Users without a value fall into
    ``unknown`` so the union of subgroups covers the full evaluated set.
    """
    lookup = pd.read_parquet(lookup_path, columns=["user_id", "age", "BiologicalSex"])
    age_by = _first_valid_by_user(lookup, "age")
    sex_by = _first_valid_by_user(lookup, "BiologicalSex")
    subgroups = {
        u: {"age_group": _age_group(age_by.get(u)), "sex": _sex(sex_by.get(u))}
        for u in {str(x) for x in users}
    }
    out_dir = Path(predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "_subgroups.json").open("w") as f:
        json.dump(subgroups, f)
