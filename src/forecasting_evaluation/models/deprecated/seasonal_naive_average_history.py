"""Seasonal naive baseline using averaged historical seasonal cycles.

.. deprecated::
    **Removed from the Track-3 leaderboard.** This baseline is no longer wired
    into the active model registry / config schema; it is archived here for
    reference only.

    Why it was dropped: the original implementation averaged the historical
    seasons with plain ``np.mean`` and selected "valid" seasons with a ``!= 0``
    test (which treats ``NaN`` as data, since ``NaN != 0`` is ``True``). With
    gappy wearable data, a single missing value anywhere in a channel's history
    propagated through ``np.mean`` and turned the whole forecast position into
    ``NaN``. ~75-99% of its predictions became ``NaN``, those rows were silently
    dropped at the skill/rank aggregation step, and the model was scored on a
    small, self-selected subset of pristine-history users — producing a
    spuriously high skill score on a non-representative cohort.

    The averaging is fixed below (``np.nanmean`` + a finite-aware season check)
    so the archived model computes what its name implies — a per-(channel,
    hour-of-day) average over the historical days that actually have data — but
    it remains deprecated and is not part of the evaluation surface.
"""

import random
import warnings

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

            # Keep a season only if it carries actual (finite, non-zero) data.
            # ``NaN != 0`` is ``True``, so the original ``!= 0`` test wrongly
            # admitted all-NaN seasons; gating on ``isfinite`` avoids that.
            if np.any(np.isfinite(candidate_season) & (candidate_season != 0)):
                valid_seasons.append(candidate_season)

            offset += 1

        if valid_seasons:
            # ``nanmean`` averages each (channel, hour) over the historical days
            # that have a value there, instead of letting one missing day poison
            # the whole position (the bug that prompted deprecation).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                averaged_season = np.nanmean(np.stack(valid_seasons, axis=0), axis=0)
        else:
            averaged_season = history[:, -effective_seasonal:]

        for k in range(prediction_length):
            idx = k % effective_seasonal
            predictions[:, k] = averaged_season[:, idx]

        point_result = predictions

        return point_result, quantiles_result
