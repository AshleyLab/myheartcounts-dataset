"""Shared machinery for public forecaster wrappers.

Each public wrapper loads a released checkpoint and exposes the
:class:`openmhc._protocols.Forecaster` interface (``predict(history,
horizon)``). The prediction itself is delegated to the corresponding internal
engine in ``forecasting_evaluation.models``, which now shares the same unified
``predict(history, horizon)`` contract — so this base class is a thin pass-through
that unwraps the point forecast.
"""

from __future__ import annotations

import numpy as np

from openmhc.forecasters._release import ReleaseLoadableMixin


class BaseForecaster(ReleaseLoadableMixin):
    """Base class for released forecasters.

    Subclasses set the class attribute ``model_name`` (matching the manifest
    ``kind``) and assign the loaded internal engine to ``self._model`` in their
    constructor. ``self._model`` exposes ``predict(history, horizon) ->
    (point, quantiles)`` returning point forecasts in the original value space.
    """

    model_name: str = ""

    _model = None  # set by subclasses

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours from the full-prefix ``history``.

        Args:
            history: Float array of shape ``(n_channels, history_length)``.
            horizon: Number of future hours to predict.

        Returns:
            Float32 array of shape ``(n_channels, horizon)``.
        """
        point, _quantiles = self._model.predict(np.asarray(history, dtype=np.float32), int(horizon))
        return np.asarray(point, dtype=np.float32)
