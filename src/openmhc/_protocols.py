"""Duck-typed protocols for MHC-Benchmark model interfaces.

Users implement these interfaces to plug custom models into the benchmark.
No base class inheritance required -- just implement the right methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """Protocol for health-prediction encoders.

    The model contract for Surface 1 (external submissions) and the bundled
    Surface-2 baselines alike.

    An encoder maps **one participant's eligible wearable data** — already filtered
    to the task's cohort and temporal window by the benchmark (IC + TC) — to a
    single fixed-size embedding. The benchmark then fits a *uniform* linear probe
    on the embeddings and scores it, so a model's result reflects its representation
    rather than its choice of probe. The same protocol is implemented by external
    encoders and by the baselines (MAE, SSL, Toto, Chronos-2, ...).

    Set ``input_granularity`` to choose how the benchmark hands you each
    participant's data. ``"daily"`` is currently wired by the segment binder:

      - ``"daily"``  — eligible daily segments ``(n_segments, 24, 38)``.
      - ``"series"`` / ``"weekly"`` — planned (hourly series / weekly segments).

    Channels 0-18 are the **raw** sensor values (NaN at missing positions) and 19-37
    the missingness mask (1 = missing, 0 = observed). Normalization is your model's
    concern — z-score with your own train-split statistics if you need it (the
    imputation track hands raw values the same way). The benchmark fits a PCA-50 probe
    on the embeddings, so return at least 50 dimensions.

    Example:
        >>> class MyEncoder:
        ...     input_granularity = "daily"
        ...     def encode(self, data: np.ndarray) -> np.ndarray:
        ...         # data: (n_segments, 24, 38) — eligible days; channels 0-18 raw
        ...         # values (NaN at missing), 19-37 the mask. Return D >= 50.
        ...         x = np.nan_to_num(data).reshape(-1, 38)
        ...         return np.concatenate([x.mean(0), x.std(0)])   # -> (76,)
    """

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Map one participant's eligible data to a 1-D float32 embedding.

        Args:
            data: the participant's eligible data at ``input_granularity`` (see the
                class docstring). Only in-cohort, in-window data is included, so it
                may be pooled / windowed freely.

        Returns:
            A 1-D float32 embedding ``(D,)`` of any dimensionality.
        """
        ...


@runtime_checkable
class Predictor(Protocol):
    """Protocol for end-to-end prediction models that own their classifier head.

    Unlike :class:`Encoder` (which returns a representation for the benchmark's
    uniform probe), a ``Predictor`` fits on the cohort's eligible data + labels and
    returns one prediction per participant; the benchmark scores those directly.
    Used by end-to-end models (e.g. GRU-D, MultiRocket). Set ``input_granularity``
    as for :class:`Encoder` (default ``"series"``).

    This is the *public* contract — plain per-participant arrays in, predictions out.
    The benchmark adapts it to its internal engine. (Bundled baselines implement the
    richer internal model interface directly, so they can key by user / task.)

    Example:
        >>> class MyPredictor:
        ...     def fit(self, data, labels): ...            # data: list of per-participant arrays
        ...     def predict(self, data) -> np.ndarray: ...   # (n,) predictions, aligned with data
    """

    def fit(self, data: list[np.ndarray], labels: np.ndarray) -> None:
        """Fit on the train cohort: per-participant eligible data + aligned labels."""
        ...

    def predict(self, data: list[np.ndarray]) -> np.ndarray:
        """Return one prediction per participant, aligned with ``data``."""
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for imputation evaluation.

    Example:
        >>> class MeanImputer:
        ...     def fit(self, data, masks):
        ...         self.means = np.nanmean(data, axis=(0, 2))
        ...     def impute(self, data, observed_mask, target_mask):
        ...         result = data.copy()
        ...         for ch in range(19):
        ...             target = target_mask[:, ch, :] == 1
        ...             result[:, ch, :][target] = self.means[ch]
        ...         return result
    """

    def fit(self, data: np.ndarray, masks: np.ndarray) -> None:
        """Fit on training data.

        Args:
            data: Daily sensor values of shape (N, 19, 1440). NaN at missing
                positions.
            masks: Binary masks of shape (N, 19, 1440). 1 = observed,
                0 = missing.
        """
        ...

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Impute artificially masked positions.

        Args:
            data: Sensor values of shape (N, 19, 1440) with NaN at masked
                positions.
            observed_mask: Binary mask of shape (N, 19, 1440). 1 = originally
                observed, 0 = naturally missing.
            target_mask: Binary mask of shape (N, 19, 1440). 1 = positions to
                impute (a subset of observed_mask).

        Returns:
            Array of shape (N, 19, 1440) with imputed values at target
            positions. Must be float32.
        """
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecasting evaluation (Track 3).

    Forecast future hours from a history window. The benchmark evaluates
    point predictions; quantile forecasts are optional via a return signature
    extension (see below).

    Example:
        >>> class LastValueForecaster:
        ...     def predict(self, history, horizon):
        ...         # history: (n_channels, history_length)
        ...         # returns: (n_channels, horizon)
        ...         last = history[:, -1:]
        ...         return np.tile(last, (1, horizon))
    """

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours given the history window.

        Args:
            history: Float array of shape ``(n_channels, history_length)``
                with the past observations. May contain NaN at missing
                positions.
            horizon: Number of future hours to predict.

        Returns:
            Float array of shape ``(n_channels, horizon)`` with the point
            forecast. Must be float32.
        """
        ...
