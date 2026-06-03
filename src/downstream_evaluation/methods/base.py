"""Method protocol + prediction container for the Surface-2 downstream eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TaskPrediction:
    """Per-task test-set predictions produced by a Method.

    Attributes:
        y_true: (N,) ground-truth labels.
        y_pred: (N,) predictions — class **probability** for binary tasks
            (so AUPRC can be computed), or the **point prediction** (ordinal
            level / regressed value) otherwise.
        user_ids: (N,) user IDs aligned with y_true / y_pred.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    user_ids: np.ndarray


@runtime_checkable
class Method(Protocol):
    """A downstream method: produce per-task test predictions from baked features.

    Implementations load the pre-baked ``features/<name>/<split>.parquet`` tables
    (cohort + temporal scope + labels already applied) and return one
    ``TaskPrediction`` per task. All cohort/temporal logic lives private-side in
    ``convert_features.py``; methods here are pure load → model → predict.
    """

    name: str

    def predict(self, tasks: list[str]) -> dict[str, TaskPrediction]:
        """Return ``{task: TaskPrediction}`` on the test split for each task."""
        ...
