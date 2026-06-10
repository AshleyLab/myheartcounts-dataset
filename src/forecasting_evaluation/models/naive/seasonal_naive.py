"""Seasonal naive baseline forecasting model."""

import random
import warnings

import numpy as np

from forecasting_evaluation.models.base import BasePredictionModel


class SeasonalNaiveModel(BasePredictionModel):
    """Seasonal naive model using the latest non-empty seasonal cycle."""

    def __init__(
        self,
        seed: int = 42,
        seasonal: int = 24,
        max_lookback_seasons: int = 7,
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
            max_lookback_seasons: Maximum number of seasonal cycles to search back
                when the most recent season is missing (NaN) at a given hour-of-day.
            quantile_levels: Quantile levels returned alongside point forecasts.
        """
        # Set random seeds for reproducibility
        self.seed = seed
        self.seasonal = seasonal
        self.max_lookback_seasons = max_lookback_seasons
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

        # Per-channel terminal fallback: the user's own historical mean for that
        # channel. Finite wherever any history exists (always true at scored cells,
        # where the target is present). A channel with no history at all degrades to
        # 0.0 (the global mean in z-scored space) so a finite value is guaranteed.
        with warnings.catch_warnings():
            # All-NaN channels legitimately produce a RuntimeWarning here.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            channel_mean = np.nanmean(history, axis=1)
        channel_mean = np.where(np.isfinite(channel_mean), channel_mean, 0.0)

        # NaN-aware seasonal cascade. For each forecast position k, take the value at
        # the same hour-of-day one season back (t-24); if NaN, walk further back
        # (t-48, t-72, ...) up to ``max_lookback_seasons`` and use the first finite
        # value. If every seasonal lag is missing, fall back to the channel mean.
        n_seasons = min(self.max_lookback_seasons, history_length // effective_seasonal)
        for k in range(prediction_length):
            phase = k % effective_seasonal
            filled = np.zeros(n_features, dtype=bool)
            for j in range(1, n_seasons + 1):
                src = history_length - j * effective_seasonal + phase
                if src < 0:
                    break
                candidate = history[:, src]
                take = ~filled & np.isfinite(candidate)
                predictions[take, k] = candidate[take]
                filled |= take
                if filled.all():
                    break
            predictions[~filled, k] = channel_mean[~filled]

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
        # Drop only seasons with no finite observations at all; per-position
        # finiteness is enforced in the loop below. Mirrors the point cascade:
        # NaN — not zero — marks missingness, so legitimately-zero hours remain in
        # the empirical pool instead of being discarded as "empty".
        valid_season_mask = np.any(np.isfinite(seasons), axis=(0, 2))
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
