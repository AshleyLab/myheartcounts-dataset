"""Typed data containers for forecasting evaluation data flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SubTrajectoryInput:
    """Container for a single forecasting sub-trajectory.

    .. deprecated::
        Produced only by the deprecated :class:`SubTrajectoryGenerator`. The
        evaluation main path no longer builds this container; it slices windows
        directly from the raw ``history_cf`` cache. Kept for reference only.

    Attributes:
        history: Historical target values with shape (n_features, history_length).
        history_mask: Optional observed-value mask for history with shape
            (n_features, history_length). True indicates observed.
        variable_names: Names of target variables/channels.
        past_covariates: Optional past-only covariates with shape (history_length,) per key.
        future_covariates: Optional covariates available for both history and forecast horizon
            with shape (history_length + prediction_hours,) per key.
        static_covariates: Optional user-level covariates (non-temporal), kept as a
            placeholder interface for model-specific encoding.
        ground_truth: Future target values with shape (n_features, prediction_hours).
        ground_truth_mask: Optional observed-value mask for forecast horizon with shape
            (n_features, prediction_hours). True indicates observed.
        index_days: Sliding index position (in days) where forecast starts.
        prediction_hours: Forecast horizon in hours.
    """

    history: np.ndarray
    variable_names: list[str]
    past_covariates: dict[str, np.ndarray] | None
    future_covariates: dict[str, np.ndarray] | None
    static_covariates: dict[str, Any] | None
    ground_truth: np.ndarray
    index_days: int
    prediction_hours: int
    history_mask: np.ndarray | None = None
    ground_truth_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate shape consistency for downstream model/evaluation logic."""
        if self.history.ndim != 2:
            raise ValueError("history must be 2D with shape (n_features, history_length)")
        if self.ground_truth.ndim != 2:
            raise ValueError("ground_truth must be 2D with shape (n_features, prediction_hours)")

        n_features, history_length = self.history.shape
        if self.ground_truth.shape[0] != n_features:
            raise ValueError("history and ground_truth must have the same number of features")
        if self.ground_truth.shape[1] != self.prediction_hours:
            raise ValueError("ground_truth second dimension must match prediction_hours")

        if len(self.variable_names) != n_features:
            raise ValueError("variable_names length must match number of history features")

        if self.history_mask is not None:
            if self.history_mask.shape != self.history.shape:
                raise ValueError("history_mask shape must match history shape")

        if self.ground_truth_mask is not None:
            if self.ground_truth_mask.shape != self.ground_truth.shape:
                raise ValueError("ground_truth_mask shape must match ground_truth shape")

        # Normalize optional covariates to empty dicts for downstream iteration safety.
        if self.past_covariates is None:
            self.past_covariates = {}
        if self.future_covariates is None:
            self.future_covariates = {}

        for key, values in self.past_covariates.items():
            if values.shape[0] != history_length:
                raise ValueError(
                    f"past_covariates['{key}'] length {values.shape[0]} does not match history length {history_length}"
                )

        expected_future_length = history_length + self.prediction_hours
        for key, values in self.future_covariates.items():
            if values.shape[0] != expected_future_length:
                raise ValueError(
                    f"future_covariates['{key}'] length {values.shape[0]} does not match expected length {expected_future_length}"
                )
