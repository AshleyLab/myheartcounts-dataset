"""Seasonal naive baseline using averaged historical seasonal cycles."""

import random

import numpy as np

from forecasting_evaluation.models.base import BasePredictionModel


class SeasonalNaiveAverageModel(BasePredictionModel):
    """Seasonal naive model that averages over multiple historical seasons."""

    def __init__(
        self,
        seed: int = 42,
        seasonal: int = 24,
    ):
        """Initialize model state.

        Args:
            seed: Random seed for reproducibility.
            seasonal: Seasonal cycle length in hours.
        """
        self.seed = seed
        self.seasonal = seasonal
        np.random.seed(seed)
        random.seed(seed)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Predict future values using averaged seasonal naive approach.

        For each seasonal position (e.g. each hour within a day when seasonal=24),
        this model collects values from all valid historical seasonal cycles and
        uses their mean as the prediction template.

        Args:
            history: Full-prefix history of shape (n_features, history_length),
                may contain NaN.
            horizon: Number of future hours to forecast.

        Returns:
            Tuple containing (point_result, quantiles_result):
            - point_result: (n_features, prediction_length) array of point predictions.
            - quantiles_result: None for this deterministic baseline.
        """
        point_result = None
        quantiles_result = None

        prediction_length = horizon

        n_features, history_length = history.shape
        effective_seasonal = min(self.seasonal, history_length)

        predictions = np.zeros((n_features, prediction_length))

        valid_seasons: list[np.ndarray] = []
        offset = 0

        while True:
            end_idx = history_length - effective_seasonal * offset
            start_idx = end_idx - effective_seasonal

            if start_idx < 0:
                break

            candidate_season = history[:, start_idx:end_idx]

            if np.any(candidate_season != 0):
                valid_seasons.append(candidate_season)

            offset += 1

        if valid_seasons:
            averaged_season = np.mean(np.stack(valid_seasons, axis=0), axis=0)
        else:
            averaged_season = history[:, -effective_seasonal:]

        for k in range(prediction_length):
            idx = k % effective_seasonal
            predictions[:, k] = averaged_season[:, idx]

        point_result = predictions
        
        return point_result, quantiles_result
