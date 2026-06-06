"""Shared machinery for public forecaster wrappers.

Each public wrapper loads a released checkpoint and exposes the
:class:`openmhc._protocols.Forecaster` interface (``predict(history,
horizon)``). The prediction itself is delegated to the corresponding internal
engine in ``forecasting_evaluation.models``; this base class only bridges the
public ``(history, horizon)`` signature to the internal
``SubTrajectoryInput`` contract and unwraps the point forecast.
"""

from __future__ import annotations

import numpy as np

from openmhc.forecasters._release import ReleaseLoadableMixin


def _to_subtrajectory(history: np.ndarray, horizon: int):
    """Wrap a ``(n_channels, history_length)`` window as a ``SubTrajectoryInput``.

    The internal models only read ``history``, the covariates and
    ``prediction_hours`` — ``ground_truth`` is required by the dataclass
    validator but unused at inference, so we pass a zero placeholder.
    """
    from forecasting_evaluation.data.types import SubTrajectoryInput

    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2:
        raise ValueError(
            f"history must be 2D (n_channels, history_length); got shape {history.shape}"
        )
    n_channels = history.shape[0]
    return SubTrajectoryInput(
        history=history,
        variable_names=[str(i) for i in range(n_channels)],
        past_covariates=None,
        future_covariates=None,
        static_covariates=None,
        ground_truth=np.zeros((n_channels, horizon), dtype=np.float32),
        index_days=0,
        prediction_hours=int(horizon),
    )


class BaseForecaster(ReleaseLoadableMixin):
    """Base class for released forecasters.

    Subclasses set the class attribute ``model_name`` (matching the manifest
    ``kind``) and assign the loaded internal engine to ``self._model`` in their
    constructor. ``self._model`` must expose ``predict(SubTrajectoryInput) ->
    (point, quantiles)`` returning point forecasts in the original value space.
    """

    model_name: str = ""

    _model = None  # set by subclasses

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours from ``history``.

        Args:
            history: Float array of shape ``(n_channels, history_length)``.
            horizon: Number of future hours to predict.

        Returns:
            Float32 array of shape ``(n_channels, horizon)``.
        """
        point, _quantiles = self._model.predict(_to_subtrajectory(history, horizon))
        return np.asarray(point, dtype=np.float32)
