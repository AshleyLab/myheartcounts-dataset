"""Sub-trajectory generator for forecasting evaluation.

Generates sub-trajectories with sliding index and valid data checking.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from data.transforms.nan_transforms import ZeroToNaNTransform
from forecasting_evaluation.data.types import SubTrajectoryInput

logger = logging.getLogger(__name__)

SAMPLE_INDEX_REQUIRED_MSG = (
    "Forecasting evaluation requires a precomputed sample index. "
    "Please generate it first with scripts/precompute_forecasting_inputs.py."
)


class SubTrajectoryGenerator:
    """Generate day-sliding forecasting samples from extracted trajectory features."""

    def __init__(
        self,
        prediction_hours: int,
        random_seed: int = 42,
        pre_sample_path: str = "",
        daily_start_hour_offset: int = 0,
    ):
        """Initialize generator with forecasting configuration.

        Args:
            prediction_hours: Forecast horizon in hours.
            random_seed: Seed used for deterministic sampling-related behavior.
            pre_sample_path: JSON path containing precomputed day indices per user.
            daily_start_hour_offset: Runtime hour offset applied to each
                precomputed day boundary.
        """
        self.prediction_hours = int(prediction_hours)
        self.hours_per_day = 24
        self.pre_sample_path = pre_sample_path
        self.random_seed = random_seed
        self.daily_start_hour_offset = int(daily_start_hour_offset)
        self._pre_sample_index: dict[str, list[int]] | None = None

        if not 0 <= self.daily_start_hour_offset < self.hours_per_day:
            raise ValueError(
                "daily_start_hour_offset must be within [0, 24), "
                f"but received {self.daily_start_hour_offset}"
            )

        self.zero_to_nan_transform = ZeroToNaNTransform()

        np.random.seed(self.random_seed)

    def _load_pre_sample_index(self) -> None:
        """Load pre-generated sample indices from JSON, once per generator instance."""
        if self._pre_sample_index is not None:
            return

        self._pre_sample_index = {}
        if not self.pre_sample_path:
            raise ValueError(SAMPLE_INDEX_REQUIRED_MSG)

        sample_path = Path(self.pre_sample_path)
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample index file not found: {sample_path}. {SAMPLE_INDEX_REQUIRED_MSG}"
            )

        loaded = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            logger.warning("Sample index file format invalid (expect dict): %s", sample_path)
            return

        normalized: dict[str, list[int]] = {}
        for user_id, day_indices in loaded.items():
            if not isinstance(day_indices, list):
                continue
            parsed_days: set[int] = set()
            for day in day_indices:
                try:
                    day_int = int(day)
                except (TypeError, ValueError):
                    continue
                if day_int >= 1:
                    parsed_days.add(day_int)
            valid_days = sorted(parsed_days)
            normalized[str(user_id)] = valid_days

        self._pre_sample_index = normalized
        # logger.info("Loaded pre-sample index for %d users from %s", len(normalized), sample_path)

    def _resolve_candidate_days(self, user_id: str | None) -> list[int]:
        """Get candidate day indices for a user.

        Candidate days come from the required pre-generated sample index JSON.
        """
        self._load_pre_sample_index()
        if user_id is None:
            raise ValueError("user_id is required when resolving sample-index candidate days")
        return [day for day in self._pre_sample_index[user_id]]

    def generate(
        self,
        features: dict,
        user_id: str | None = None,
    ) -> Iterator[SubTrajectoryInput]:
        """Generate sub-trajectories for forecasting evaluation.

        Args:
            features: Feature dictionary containing:
                - history: (n_features, trajectory_length) array
                - history_mask: (n_features, trajectory_length) bool/int array
                - variable_names: list[str] with len == n_features
                - past_covariates: dict of (trajectory_length,) arrays, optional
                - future_covariates: dict of (trajectory_length + prediction_length,) arrays, optional
                - static_covariates: user-level non-temporal covariates, optional
            user_id: User identifier used to resolve candidate day indices from
                precomputed sample-index data.
        
        Yields:
            SubTrajectoryInput objects containing:
                - history and ground_truth in channel-first layout
                - history_mask and ground_truth_mask aligned with each segment
                - covariate slices aligned to the emitted history/prediction window
        """
        history = features["history"]  # (n_features, trajectory_length)
        # TODO preprocessing on minute level
        history = self.zero_to_nan_transform(torch.from_numpy(history)).numpy()

        variable_names = features["variable_names"]
        past_covariates = features.get("past_covariates") or {}
        future_covariates = features.get("future_covariates") or {}
        static_covariates = features.get("static_covariates")

        prediction_hours = self.prediction_hours
        candidate_days = self._resolve_candidate_days(user_id=user_id)

        for current_day in candidate_days:
            index_hours = current_day * self.hours_per_day + self.daily_start_hour_offset

            # Emit sample using all data before index as history.
            history_start = 0
            history_end = index_hours
            pred_end = index_hours + prediction_hours

            if history_end <= history_start:
                continue
            if pred_end > history.shape[1]:
                logger.debug(
                    "Skip sample (user=%s, day=%s): pred_end=%s exceeds trajectory_length=%s",
                    user_id,
                    current_day,
                    pred_end,
                    history.shape[1],
                )
                continue

            # Extract history target (all data before index)
            history_target = history[:, history_start:history_end]  # (n_features, index_hours)

            # Extract ground truth (future values for prediction)
            ground_truth = history[:, history_end:pred_end]  # (n_features, prediction_hours)

            # Extract past covariates (same length as history)
            sub_past_covariates = {}
            for key, values in past_covariates.items():
                sub_past_covariates[key] = values[history_start:history_end]

            # Extract future covariates (history + prediction length)
            sub_future_covariates = {}
            for key, values in future_covariates.items():
                sub_future_covariates[key] = values[history_start:pred_end]

            yield SubTrajectoryInput(
                history=history_target,
                variable_names=variable_names,
                past_covariates=sub_past_covariates,
                future_covariates=sub_future_covariates,
                static_covariates=static_covariates,
                ground_truth=ground_truth,
                index_days=current_day,
                prediction_hours=prediction_hours
            )
