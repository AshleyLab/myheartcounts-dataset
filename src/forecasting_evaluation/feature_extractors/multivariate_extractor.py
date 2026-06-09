"""Multivariate feature extractor for time series forecasting.

Extracts multivariate time series data from trajectory samples for forecasting tasks.
"""

from __future__ import annotations

import logging

import numpy as np

from forecasting_evaluation.config import FeaturesConfig

logger = logging.getLogger(__name__)


class MultivariateFeatureExtractor:
    """Extract multivariate forecasting features from one trajectory row.

    Output schema is consumed by the evaluator's raw history cache + row-group
    manifest path and includes both values and observed-value masks in
    channel-first format.
    """

    def __init__(self, config: FeaturesConfig, forecasting_length: int = 24):
        """Initialize multivariate feature extractor.

        Args:
            config: Required forecasting feature configuration.
            forecasting_length: Number of hours to forecast (for future covariates).
        """
        self.config = config
        self.prediction_length = forecasting_length

    def extract(self, trajectory: dict) -> dict:
        """Extract multivariate features from a single trajectory.

        Args:
            trajectory: Single trajectory sample with the following fields:
                - user_id: str
                - values: (T, C) float32 array - T timesteps, C channels
                - timestamps: list of length T
                - channel_names: list of length C
                - channel_units: list of length C
                - start_time: str (ISO 8601)
                - end_time: str (ISO 8601)

        Returns:
            Dictionary containing:
                {
                    "history": np.ndarray of shape (n_features, trajectory_length),
                    "variable_names": list[str],
                    "past_covariates": dict[str, np.ndarray] | None,
                    "future_covariates": dict[str, np.ndarray] | None,
                    "static_covariates": dict[str, object] | None,
                }

        Note:
                Covariates are placeholder interfaces in the current benchmark release.
                We intentionally do not implement feature engineering here; model-specific
                encoding is deferred to model implementations in later iterations.
        """
        # Extract data from trajectory
        values = np.asarray(trajectory["values"], dtype=np.float32)  # (T, C)
        # mask = np.asarray(trajectory["mask"], dtype=bool)  # (T, C)
        channel_names = trajectory["channel_names"]
        # timestamps = trajectory["timestamps"]

        channel = self.config.channel

        if channel != "all":
            raise ValueError(f"Unknown channel type: {channel}")

        # Select target channels for both values and masks.
        selected_values = values  # (T, 19)
        # selected_mask = mask[:, channel_indices]  # (T, n_features)
        selected_channel_names = list(channel_names)

        # Convert from (T, n_features) to (n_features, T) for forecasting pipeline.
        history = selected_values.T  # (n_features, T)
        # history_mask = selected_mask.T  # (n_features, T)
        # trajectory_length = history.shape[1]
        
        # Placeholder-only covariate interfaces:
        # - dynamic covariates (past/future) are kept as empty dicts by default.
        # - static covariates are set to None.
        # This preserves a stable contract without forcing a single encoding scheme.
        covariate_types = self.config.covariate_types if self.config is not None else None
        if covariate_types:
            logger.info(
                "covariate_types=%s requested, but covariate engineering is intentionally deferred; "
                "returning placeholder empty covariate dicts.",
                covariate_types,
            )

        past_covariates: dict[str, np.ndarray] = {}
        future_covariates: dict[str, np.ndarray] = {}
        static_covariates: dict[str, object] | None = None
        
        return {
            "history": history,
            "variable_names": selected_channel_names,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "static_covariates": static_covariates,
        }
