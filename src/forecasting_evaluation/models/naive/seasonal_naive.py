"""Seasonal naive baseline forecasting model."""

import random

import numpy as np

from forecasting_evaluation.models.base import BasePredictionModel


class SeasonalNaiveModel(BasePredictionModel):
    """Seasonal naive model using the latest non-empty seasonal cycle."""

    def __init__(
        self,
        seed: int = 42,
        seasonal: int = 24,
        quantile_levels: tuple[float, ...] | list[float] | np.ndarray = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
    ):
        """Initialize model state.

        Args:
            seed: Random seed for reproducibility.
            seasonal: Seasonal cycle length in hours.
            quantile_levels: Quantile levels returned alongside point forecasts.
        """
        # Set random seeds for reproducibility
        self.seed = seed
        self.seasonal = seasonal
        self.quantile_levels = self._validate_quantile_levels(quantile_levels)
        np.random.seed(seed)
        random.seed(seed)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Predict future values using seasonal naive approach.

        Args:
            history: Full-prefix history of shape (n_features, history_length),
                may contain NaN.
            horizon: Number of future hours to forecast.

        Returns:
            Tuple containing (point_result, quantiles_result):
            - point_result: (n_features, prediction_length) array of point predictions.
            - quantiles_result: (n_features, prediction_length, n_quantiles) array
              of empirical seasonal quantile forecasts.
        """
        point_result = None

        prediction_length = horizon

        n_features, history_length = history.shape

        # Ensure history_length is at least as long as seasonal period
        effective_seasonal = min(self.seasonal, history_length)

        # Initialize prediction array
        predictions = np.zeros((n_features, prediction_length))

        # Get the most recent complete seasonal period, skip if all zeros
        # Shape: (n_features, effective_seasonal)
        last_season = None
        offset = 0
        while offset * effective_seasonal < history_length:
            start_idx = max(0, history_length - effective_seasonal * (offset + 1))
            end_idx = history_length - effective_seasonal * offset
            candidate_season = history[:, start_idx:end_idx]

            # Check if all zeros
            if np.any(candidate_season != 0):
                last_season = candidate_season
                break
            offset += 1

        # If all historical data is zero, use the most recent period (even if all zeros)
        if last_season is None:
            last_season = history[:, -effective_seasonal:]

        # Adjust effective_seasonal to match the actual season length obtained
        effective_seasonal = last_season.shape[1]

        for k in range(prediction_length):
            # Use modulo operation for cyclic reference
            # e.g., k=0, 24, 48... will all map to the same position in last_season
            idx = k % effective_seasonal
            predictions[:, k] = last_season[:, idx]

        # Point predictions
        # Shape: (n_features, prediction_length)
        point_result = predictions
        quantiles_result = self._predict_empirical_quantiles(
            history=history,
            predictions=predictions,
            prediction_length=prediction_length,
            effective_seasonal=effective_seasonal,
        )

        return point_result, quantiles_result

    @staticmethod
    def _validate_quantile_levels(
        quantile_levels: tuple[float, ...] | list[float] | np.ndarray,
    ) -> np.ndarray:
        """Validate and normalize configured quantile levels."""
        quantile_array = np.asarray(quantile_levels, dtype=float)
        if quantile_array.ndim != 1 or quantile_array.size == 0:
            raise ValueError("quantile_levels must be a non-empty 1D sequence")
        if np.any((quantile_array <= 0.0) | (quantile_array >= 1.0)):
            raise ValueError("quantile_levels must be strictly between 0 and 1")
        if np.unique(quantile_array).size != quantile_array.size:
            raise ValueError("quantile_levels must not contain duplicates")
        return np.sort(quantile_array)

    def _predict_empirical_quantiles(
        self,
        *,
        history: np.ndarray,
        predictions: np.ndarray,
        prediction_length: int,
        effective_seasonal: int,
    ) -> np.ndarray:
        """Estimate quantiles from matching positions in non-empty full seasons."""
        n_features, history_length = history.shape
        n_quantiles = len(self.quantile_levels)
        quantiles = np.zeros((n_features, prediction_length, n_quantiles))

        n_complete_seasons = history_length // effective_seasonal
        if n_complete_seasons <= 0:
            return np.repeat(predictions[:, :, None], n_quantiles, axis=2)

        season_start = history_length - n_complete_seasons * effective_seasonal
        seasons = history[:, season_start:].reshape(
            n_features,
            n_complete_seasons,
            effective_seasonal,
        )
        valid_season_mask = np.any(seasons != 0, axis=(0, 2))
        valid_seasons = seasons[:, valid_season_mask, :]

        if valid_seasons.shape[1] == 0:
            return np.repeat(predictions[:, :, None], n_quantiles, axis=2)

        for k in range(prediction_length):
            idx = k % effective_seasonal
            seasonal_values_by_feature = valid_seasons[:, :, idx].astype(float, copy=False)
            for feature_idx in range(n_features):
                seasonal_values = seasonal_values_by_feature[feature_idx]
                seasonal_values = seasonal_values[np.isfinite(seasonal_values)]
                if seasonal_values.size == 0:
                    quantiles[feature_idx, k, :] = predictions[feature_idx, k]
                else:
                    quantiles[feature_idx, k, :] = np.quantile(
                        seasonal_values,
                        self.quantile_levels,
                    )

        return quantiles
