"""Demographic covariate augmentation for the downstream-eval prediction track.

Only the demographic-aware baseline (stat_simple) appends demographic columns
(``age``, ``BiologicalSex``, ``BMI_values``) onto its feature matrix before the
probe; every other method leaves them off. Two responsibilities live here:

1. **Per-task feature exclusions** — loaded once from
   ``data/labels/task_feature_exclusions.json``. When a task itself is a
   demographic (e.g. ``age``) or a closed-form function of demographics
   (e.g. ``framingham_risk``), the aliased covariates are excluded to prevent
   trivial leakage. The JSON is data-driven so rules can be added without code
   changes. Self-exclusion (a covariate whose name equals the task) is always
   applied automatically and need not be listed.
2. **Augmentation** — ``build_demo_user_lookup_from_labels_df`` builds a per-user
   ``{user_id: [cov, ...]}`` lookup from the labels lookup (which carries the
   demographic columns); ``apply_demographics`` concatenates the non-excluded
   covariates onto ``X``; ``count_demo_cols`` reports how many it would add.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task feature exclusions (DEMO_ALIAS)
# ---------------------------------------------------------------------------
# Loaded at import time so rules can be edited as data, not code. Self-exclusion
# (a covariate matching the task name) is always applied automatically and does
# not need to be listed in the JSON.

_TASK_FEATURE_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "labels" / "task_feature_exclusions.json"
)


def _load_task_feature_exclusions(path: Path) -> dict[str, tuple[str, ...]]:
    """Load per-task feature exclusion rules from JSON.

    Schema: ``{task_name: [forbidden_feature, ...]}``. Top-level keys starting
    with ``_`` (e.g. ``_doc``, ``_rationale``) are documentation and skipped.
    Raises ``FileNotFoundError`` if missing — it ships with the repo's data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Task feature exclusions JSON not found at {path}. "
            "This file ships with the repo at data/labels/task_feature_exclusions.json."
        )
    with open(path) as f:
        raw = json.load(f)
    return {task: tuple(features) for task, features in raw.items() if not task.startswith("_")}


DEMO_ALIAS: dict[str, tuple[str, ...]] = _load_task_feature_exclusions(
    _TASK_FEATURE_EXCLUSIONS_PATH
)


# ---------------------------------------------------------------------------
# Per-user demographic lookup
# ---------------------------------------------------------------------------


def build_demo_user_lookup_from_labels_df(
    labels_df,
    demo_covariates: list[str],
) -> dict[str, np.ndarray]:
    """Build ``{user_id: np.array([cov, ...])}`` from the labels lookup.

    Takes the first non-sentinel value per user, fills missing with 0.0. The
    labels lookup carries the demographic columns (``age``/``BiologicalSex``/
    ``BMI_values``), so no separate data source is needed.
    """
    _demo_df = labels_df[["user_id"] + demo_covariates].copy()
    for c in demo_covariates:
        sentinel = -1.0 if _demo_df[c].dtype in (np.float64, np.float32) else -1
        _demo_df.loc[_demo_df[c] == sentinel, c] = np.nan
    _user_demo = _demo_df.groupby("user_id")[demo_covariates].first().fillna(0.0)
    return {uid: row.values.astype(np.float32) for uid, row in _user_demo.iterrows()}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _kept_cov_indices(task_name: str, demo_covariates: list[str]) -> list[int]:
    """Indices of covariates kept for ``task_name`` (drop self + DEMO_ALIAS)."""
    return [
        i
        for i, c in enumerate(demo_covariates)
        if c != task_name and c not in DEMO_ALIAS.get(task_name, ())
    ]


def apply_demographics(
    X: np.ndarray,
    uids: np.ndarray,
    task_name: str,
    demo_user_lookup: dict[str, np.ndarray] | None,
    demo_covariates: list[str],
) -> np.ndarray:
    """Append the non-excluded demographic covariates to feature matrix ``X``.

    Covariates matching ``task_name`` (self) or listed in ``DEMO_ALIAS`` are
    excluded to prevent label leakage. Returns ``X`` unchanged when no lookup
    or covariate list is provided, or when all covariates are excluded.
    """
    if demo_user_lookup is None or not demo_covariates:
        return X
    cov_indices = _kept_cov_indices(task_name, demo_covariates)
    if not cov_indices:
        return X
    demo_matrix = np.zeros((len(uids), len(cov_indices)), dtype=np.float32)
    for row_idx, uid in enumerate(uids):
        vec = demo_user_lookup.get(uid)
        if vec is not None:
            demo_matrix[row_idx] = vec[cov_indices]
    return np.hstack([X, demo_matrix])


def count_demo_cols(
    task_name: str,
    demo_user_lookup: dict[str, np.ndarray] | None,
    demo_covariates: list[str],
) -> int:
    """Number of demographic columns ``apply_demographics`` would append."""
    if demo_user_lookup is None or not demo_covariates:
        return 0
    return len(_kept_cov_indices(task_name, demo_covariates))
