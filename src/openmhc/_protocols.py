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

    Encode weekly sensor tensors into fixed-size embeddings.

    Example:
        >>> class MyEncoder:
        ...     def encode(self, weekly_tensors: np.ndarray) -> np.ndarray:
        ...         # weekly_tensors: (B, 168, 38)
        ...         # Returns: (B, D) embeddings
        ...         return weekly_tensors[:, :, :19].mean(axis=1)
    """

    def encode(self, weekly_tensors: np.ndarray) -> np.ndarray:
        """Map weekly sensor data to fixed-size embeddings.

        Args:
            weekly_tensors: Array of shape (B, 168, 38).
                Columns 0-18 are z-scored sensor values (19 channels, hourly).
                Columns 19-37 are missingness masks (1 = missing, 0 = observed).

        Returns:
            Array of shape (B, D) where D is any embedding dimensionality.
            Must be float32.
        """
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for imputation evaluation.

    Implementations only need to define ``impute``. All setup — loading
    checkpoints, computing statistics from the training split, building
    per-user state — is the implementation's responsibility, typically
    done in ``__init__``. The benchmark harness does not call any
    preparation method.

    The optional metadata kwargs are keyword-only with ``None`` defaults.
    The harness inspects each implementation's ``impute`` signature once
    and forwards only the kwargs the implementation actually declares,
    so simple methods can keep the three-argument form below.

    Example:
        >>> class MeanImputer:
        ...     def __init__(self):
        ...         import openmhc
        ...         data_sum, data_count = 0.0, 0
        ...         for data, mask in openmhc.iter_train_data():
        ...             obs = (mask > 0.5) & np.isfinite(data)
        ...             data_sum = data_sum + np.where(obs, data, 0.0).sum(axis=(0, 2))
        ...             data_count = data_count + obs.sum(axis=(0, 2))
        ...         self.means = data_sum / np.maximum(data_count, 1)
        ...     def impute(self, data, observed_mask, target_mask):
        ...         result = data.copy()
        ...         for ch in range(19):
        ...             result[:, ch, :][target_mask[:, ch, :] == 1] = self.means[ch]
        ...         return result.astype(np.float32, copy=False)

    See ``openmhc.imputers`` for ready-to-use reference implementations
    (mean, mode, linear, LOCF, temporal, personalized, and a generic
    ``TorchImputer`` wrapper).
    """

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        user_ids: list[str] | None = None,
        dates: list[str] | None = None,
        day_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Impute artificially masked positions.

        Args:
            data: Sensor values of shape (N, 19, T) with NaN at both
                naturally missing positions and artificially masked
                positions. ``T = 1440`` for daily evaluation (the default);
                ``T = n_days * 1440`` when the caller sets ``n_days > 1``
                in ``evaluate_imputation`` (1-7).
            observed_mask: Binary mask of shape (N, 19, T). 1 =
                originally observed, 0 = naturally missing.
            target_mask: Binary mask of shape (N, 19, T). 1 = positions
                to impute (always a subset of ``observed_mask``).
            sample_indices: Optional split-local indices, shape (N,).
                Useful for any implementation that keeps per-sample state.
            user_ids: Optional list of N user-identifier strings, one per
                sample. Used by personalized methods.
            dates: Optional list of N ISO date strings (``YYYY-MM-DD``),
                one per sample.
            day_offsets: Optional int64 array of shape ``(N, n_days)``,
                only forwarded when ``n_days > 1``. Each row gives the
                calendar-day deltas of that window's day slots from the
                first non-padded day; ``-1`` marks left-padded slots that
                have no real data. Used by calendar-aware models (e.g.
                RoPE day embeddings in ``LSM2WeeklySparseImputer``) to
                encode real-world gaps between days in a non-contiguous
                window. Declaring this kwarg in your ``impute`` signature
                opts your imputer in — the harness inspects the signature
                once and only forwards declared kwargs.

        Returns:
            Array of shape (N, 19, T) with imputed values at
            ``target_mask == 1`` positions. Must be float32. ``T`` matches
            the input ``T`` (i.e. ``1440`` or ``n_days * 1440``).
        """
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecasting evaluation (Track 3) — the unified contract.

    A forecaster receives, for each in-scope window, the **full-prefix** history
    ``(n_channels, history_length)`` (selected by data-quality criteria only, so
    the window set is identical across all models) and the forecast ``horizon``.
    The model owns all context windowing / truncation / padding it needs.

    The harness **never** drops a window for model-capability reasons. If the
    model cannot predict a given window/channel/timestep it must emit ``NaN``
    there; the harness substitutes the Seasonal-Naive baseline for those
    positions before scoring and reports how often that happened
    (``ForecastingResults.overall_fallback_rate``).

    Optional metadata kwargs are keyword-only and forwarded only if declared
    (the harness inspects the signature once, the same duck-typed pattern as
    :class:`Encoder` / :class:`Imputer`): ``variable_names``,
    ``past_covariates``, ``future_covariates``, ``index_days``.

    The benchmark ranks point forecasts; quantile forecasts are optional by
    returning a ``(point, quantiles)`` tuple instead of a bare point array
    (``quantiles`` shape ``(n_channels, horizon, n_quantiles)``; expose the
    matching levels as a ``quantile_levels`` attribute).

    Example:
        >>> class LastValueForecaster:
        ...     def predict(self, history, horizon):
        ...         # history: (n_channels, history_length), full prefix
        ...         # returns: (n_channels, horizon)
        ...         last = history[:, -1:]
        ...         return np.tile(last, (1, horizon))
    """

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours given the full-prefix history.

        Args:
            history: Float array of shape ``(n_channels, history_length)`` with
                the past observations (full prefix up to the forecast origin).
                May contain NaN at missing positions and may be short.
            horizon: Number of future hours to predict.

        Returns:
            Float array of shape ``(n_channels, horizon)`` with the point
            forecast (float32), or a ``(point, quantiles)`` tuple. Use ``NaN``
            for any position the model cannot predict — those are filled by the
            Seasonal-Naive baseline before scoring.
        """
        ...
