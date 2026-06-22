"""Per-user prediction-pair substrate for the Track-1 leaderboard.

Track 1's headline metrics (binary AUPRC, ordinal Spearman, regression Pearson)
are cohort-level ranking / correlation metrics that do **not** decompose into a
per-user error the way imputation's per-user MAE does. So — unlike Tracks 2/3,
which ship a precomputed ``E_per_user`` — the Track-1 substrate ships the **raw
per-user prediction pairs** and the leaderboard recomputes paired skill / rank /
fairness server-side (vs. the Linear baseline) from them.

This module pools the per-(method, task) ``test.parquet`` files written by
``predictions_io.write_task_predictions`` into one long frame per method, expanding
each user across the ``all`` / ``age_group`` / ``sex`` subgroups (the fairness axes,
read from ``_subgroups.json``). One row per
``(method, task, task_type, subgroup_attr, subgroup_value, user_id)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from downstream_evaluation.evaluation.metrics import get_task_type
from downstream_evaluation.evaluation.predictions_io import _safe_task

PER_USER_PAIRS_PARQUET_COLUMNS = [
    "method",
    "task",
    "task_type",
    "subgroup_attr",
    "subgroup_value",
    "user_id",
    "y_true",
    "y_pred",
    "y_proba",
]

# Fairness subgroup axes, expanded alongside the global "all" cell.
_SUBGROUP_ATTRS = ("age_group", "sex")


def build_per_user_pairs(
    predictions_dir: str | Path,
    method_dir: str,
    tasks,
    *,
    subgroups_path: str | Path | None = None,
    method_label: str | None = None,
) -> pd.DataFrame:
    """Pool one method's per-(task) test predictions into the long pairs substrate.

    Args:
        predictions_dir: directory holding ``<method_dir>/<task>/test.parquet`` (as
            written by :func:`predictions_io.write_task_predictions`) and the shared
            ``_subgroups.json``.
        method_dir: the on-disk method subdirectory (``model.name``).
        tasks: the task names that were evaluated — used to recover the original
            task name from its filesystem-safe directory (the mangling is one-way).
        subgroups_path: override for the ``_subgroups.json`` location; defaults to
            ``<predictions_dir>/_subgroups.json``.
        method_label: value for the ``method`` column (the leaderboard groups by it
            and the upload filename must match it); defaults to ``method_dir``.

    Returns:
        DataFrame with :data:`PER_USER_PAIRS_PARQUET_COLUMNS`. Empty (with those
        columns) when no task files are found.
    """
    predictions_dir = Path(predictions_dir)
    method_label = method_label or method_dir
    subgroups_path = Path(subgroups_path) if subgroups_path else predictions_dir / "_subgroups.json"
    sub_map = json.loads(subgroups_path.read_text()) if subgroups_path.exists() else {}

    frames = []
    for task in tasks:
        pq = predictions_dir / method_dir / _safe_task(task) / "test.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq).rename(columns={"uid": "user_id"})
        df["user_id"] = df["user_id"].astype(str)
        df["method"] = method_label
        df["task"] = task
        df["task_type"] = get_task_type(task)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=PER_USER_PAIRS_PARQUET_COLUMNS)

    base = pd.concat(frames, ignore_index=True)

    # Subgroup expansion: the global "all" cell plus one row per fairness axis. Each
    # user's pair is duplicated under its age band and sex so the server-side fairness
    # reducer can slice without a separate subgroup map (mirrors the Track-2 substrate).
    expanded = [base.assign(subgroup_attr="all", subgroup_value="all")]
    for attr in _SUBGROUP_ATTRS:
        values = base["user_id"].map(lambda u, a=attr: sub_map.get(u, {}).get(a, "unknown"))
        expanded.append(base.assign(subgroup_attr=attr, subgroup_value=values))
    result = pd.concat(expanded, ignore_index=True)
    return result[PER_USER_PAIRS_PARQUET_COLUMNS]


def write_per_user_pairs_parquet(
    df: pd.DataFrame, path: str | Path, meta: dict | None = None
) -> None:
    """Serialize the substrate (string keys → category, pairs → float32, zstd).

    When ``meta`` is given, also writes the ``<path>.meta.json`` provenance sidecar
    (e.g. ``method`` / ``baseline`` / ``overall_fallback_rate``).
    """
    path = Path(path)
    df = df.copy()
    for col in ("method", "task", "task_type", "subgroup_attr", "subgroup_value", "user_id"):
        df[col] = df[col].astype("category")
    for col in ("y_true", "y_pred", "y_proba"):
        df[col] = df[col].astype("float32")
    df[PER_USER_PAIRS_PARQUET_COLUMNS].to_parquet(path, index=False, compression="zstd")
    if meta is not None:
        Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2))


def read_per_user_pairs_parquet(path: str | Path) -> tuple[pd.DataFrame, dict | None]:
    """Load the substrate parquet and its ``<path>.meta.json`` sidecar (if present)."""
    path = Path(path)
    df = pd.read_parquet(path)
    meta_path = Path(f"{path}.meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return df, meta
