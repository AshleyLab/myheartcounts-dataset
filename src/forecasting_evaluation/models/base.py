"""Base class for forecasting models — the unified Forecaster contract.

Forecasting models implement a single method::

    def predict(self, history, horizon, *, <optional declared kwargs>)
        -> np.ndarray | tuple[np.ndarray, np.ndarray | None]

where ``history`` is the **full-prefix** window of shape
``(n_channels, history_length)`` (data-quality-filtered, may contain NaN and may
be shorter than any fixed context the model wants), ``horizon`` is the number of
future hours, and the return is the point forecast ``(n_channels, horizon)`` —
optionally a ``(point, quantiles)`` tuple. A model MUST emit ``NaN`` for any
window/channel/timestep it cannot predict; the harness never drops a window for
model-capability reasons (it substitutes the Seasonal-Naive baseline for NaN
positions and reports how often that happened).

Inheritance is **not** required: the evaluator duck-types ``predict`` and uses
``inspect.signature`` to forward only the optional metadata kwargs a model
declares (e.g. ``variable_names``, ``past_covariates``, ``future_covariates``,
``index_days``). This base only offers conveniences: a default no-op ``reset()``
(called once per user to clear cross-user state) and the optional ``model_name``
/ ``quantile_levels`` attributes the evaluator reads via ``getattr``.
"""

from __future__ import annotations

import numpy as np


class BasePredictionModel:
    """Optional convenience base for forecasting models (see module docstring)."""

    model_name: str = ""
    quantile_levels: np.ndarray | None = None

    def reset(self) -> None:
        """Reset model state between users if applicable (default: no-op)."""
        pass
