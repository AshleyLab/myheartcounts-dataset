"""Public wrappers for the from-scratch neural forecasters (PyPOTS-backed).

DLinear, SegRNN, and MixLinear are trained on the MHC training split and
serialized as ``.pypots`` checkpoints. The released bundle is a *directory*
that co-locates the ``.pypots`` file, the ``training_config.json`` (the source
of truth for architecture), and the ``standard_scaler_stats.json`` used to
inverse-transform predictions back to real units. The internal models read all
three from that directory, so these wrappers are thin.

Requires ``pip install 'openmhc[pypots]'`` (the ``pypots`` package).
"""

from __future__ import annotations

from pathlib import Path

from openmhc.forecasters._base import BaseForecaster


class _NeuralForecaster(BaseForecaster):
    """Shared constructor for the PyPOTS-backed neural forecasters.

    ``model_path`` is the release bundle directory holding the ``.pypots``
    checkpoint plus ``training_config.json`` and ``standard_scaler_stats.json``.
    The internal model reads architecture from ``training_config.json`` (falling
    back to config defaults) and the scaler from the co-located stats file.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        normalization_stats_path: str | Path | None = None,
        device: str = "cuda",
        **arch,
    ) -> None:
        from forecasting_evaluation.config import (
            FeaturesConfig,
            ForecastingConfig,
        )

        config = self._build_config(checkpoint_path=str(model_path), device=device)
        # Architecture is sourced primarily from the bundled training_config.json;
        # any manifest ``arch`` keys that name real config fields act as fallbacks.
        for key, value in arch.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self._model = self._build_model(
            config=config,
            forecasting_config=ForecastingConfig(),
            features_config=FeaturesConfig(),
        )

    @staticmethod
    def _build_config(checkpoint_path: str, device: str):
        raise NotImplementedError

    @staticmethod
    def _build_model(config, forecasting_config, features_config):
        raise NotImplementedError


class DLinearForecaster(_NeuralForecaster):
    """Released DLinear forecaster."""

    model_name = "dlinear"

    @staticmethod
    def _build_config(checkpoint_path: str, device: str):
        from forecasting_evaluation.config import DLinearModelConfig

        return DLinearModelConfig(checkpoint_path=checkpoint_path, device=device)

    @staticmethod
    def _build_model(config, forecasting_config, features_config):
        from forecasting_evaluation.models.deep_learning_model.dlinear import DLinearModel

        return DLinearModel(
            config=config,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )


class SegRNNForecaster(_NeuralForecaster):
    """Released SegRNN forecaster."""

    model_name = "segrnn"

    @staticmethod
    def _build_config(checkpoint_path: str, device: str):
        from forecasting_evaluation.config import SegRNNModelConfig

        return SegRNNModelConfig(checkpoint_path=checkpoint_path, device=device)

    @staticmethod
    def _build_model(config, forecasting_config, features_config):
        from forecasting_evaluation.models.deep_learning_model.segrnn import SegRNNModel

        return SegRNNModel(
            config=config,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )


class MixLinearForecaster(_NeuralForecaster):
    """Released MixLinear forecaster."""

    model_name = "mixlinear"

    @staticmethod
    def _build_config(checkpoint_path: str, device: str):
        from forecasting_evaluation.config import MixLinearModelConfig

        return MixLinearModelConfig(checkpoint_path=checkpoint_path, device=device)

    @staticmethod
    def _build_model(config, forecasting_config, features_config):
        from forecasting_evaluation.models.deep_learning_model.mixlinear import MixLinearModel

        return MixLinearModel(
            config=config,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )
