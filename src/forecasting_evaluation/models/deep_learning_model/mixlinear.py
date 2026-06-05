"""Evaluator wrapper for a trained PyPOTS MixLinear checkpoint."""

from __future__ import annotations

from forecasting_evaluation.config import FeaturesConfig, ForecastingConfig, MixLinearModelConfig
from forecasting_evaluation.models.deep_learning_model.pypots_forecasting_base import (
    BasePyPOTSForecastingModel,
    infer_n_features,
)


class MixLinearModel(BasePyPOTSForecastingModel):
    """Evaluator adapter for PyPOTS MixLinear."""

    def __init__(
        self,
        config: MixLinearModelConfig,
        forecasting_config: ForecastingConfig,
        features_config: FeaturesConfig,
    ) -> None:
        """Initialize the evaluator adapter around a saved MixLinear checkpoint."""
        if not config.checkpoint_path:
            raise ValueError(
                "MixLinear forecasting requires model.release_dir or "
                "model.mixlinear.checkpoint_path."
            )
        self.config = config
        self.forecasting_config = forecasting_config
        self.features_config = features_config
        super().__init__(checkpoint_path=config.checkpoint_path, model_name="mixlinear")

    def build_model(self):
        """Instantiate the PyPOTS MixLinear model for inference."""
        from pypots.forecasting import MixLinear

        n_pred_steps = (
            self._get_training_config_value("model", "n_pred_steps")
            or self.config.n_pred_steps
            or self.forecasting_config.forecasting_length
        )
        n_features = (
            self._get_training_config_value("model", "n_features")
            or self.config.n_features
            or infer_n_features(self.features_config)
        )
        period_len = self._get_training_config_value("model", "period_len")
        lpf = self._get_training_config_value("model", "lpf")
        alpha = self._get_training_config_value("model", "alpha")
        rank = self._get_training_config_value("model", "rank")
        batch_size = self._get_training_config_value("training", "batch_size")
        device = self._get_training_config_value("training", "device")

        return MixLinear(
            n_steps=self.n_steps,
            n_features=n_features,
            n_pred_steps=n_pred_steps,
            n_pred_features=n_features,
            period_len=self.config.period_len if period_len is None else period_len,
            lpf=self.config.lpf if lpf is None else lpf,
            alpha=self.config.alpha if alpha is None else alpha,
            rank=self.config.rank if rank is None else rank,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            device=self.config.device if device is None else device,
            saving_path=None,
        )

    @property
    def n_steps(self) -> int:
        """Return the fixed history length expected by MixLinear."""
        return (
            self._get_training_config_value("model", "n_steps")
            or self.config.n_steps
            or (_raise_missing_n_steps())
        )

    @property
    def n_pred_steps(self) -> int:
        """Return the forecast horizon expected by MixLinear."""
        return (
            self._get_training_config_value("model", "n_pred_steps")
            or self.config.n_pred_steps
            or self.forecasting_config.forecasting_length
        )

    @property
    def n_features(self) -> int:
        """Return the channel count expected by MixLinear."""
        return (
            self._get_training_config_value("model", "n_features")
            or self.config.n_features
            or infer_n_features(self.features_config)
        )


def _raise_missing_n_steps() -> int:
    raise ValueError(
        "Unable to infer evaluation history length for MixLinear. "
        "Set model.mixlinear.n_steps in the eval config or use a checkpoint that includes "
        "training_config.json/training_config.yaml."
    )
