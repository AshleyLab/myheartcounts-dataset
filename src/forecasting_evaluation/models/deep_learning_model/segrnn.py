"""Evaluator wrapper for a trained PyPOTS SegRNN checkpoint."""

from __future__ import annotations

from forecasting_evaluation.config import FeaturesConfig, ForecastingConfig, SegRNNModelConfig
from forecasting_evaluation.models.deep_learning_model.pypots_forecasting_base import (
    BasePyPOTSForecastingModel,
    infer_n_features,
)


class SegRNNModel(BasePyPOTSForecastingModel):
    """Evaluator adapter for PyPOTS SegRNN."""

    def __init__(
        self,
        config: SegRNNModelConfig,
        forecasting_config: ForecastingConfig,
        features_config: FeaturesConfig,
    ) -> None:
        """Initialize the evaluator adapter around a saved SegRNN checkpoint."""
        if not config.checkpoint_path:
            raise ValueError(
                "SegRNN forecasting requires model.release_dir or "
                "model.segrnn.checkpoint_path."
            )
        self.config = config
        self.forecasting_config = forecasting_config
        self.features_config = features_config
        super().__init__(checkpoint_path=config.checkpoint_path, model_name="segrnn")

    def build_model(self):
        """Instantiate the PyPOTS SegRNN model for inference."""
        from pypots.forecasting import SegRNN
        from pypots.optim import Adam

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
        seg_len = self._get_training_config_value("model", "seg_len")
        d_model = self._get_training_config_value("model", "d_model")
        dropout = self._get_training_config_value("model", "dropout")
        batch_size = self._get_training_config_value("training", "batch_size")
        device = self._get_training_config_value("training", "device")

        return SegRNN(
            n_steps=self.n_steps,
            n_features=n_features,
            n_pred_steps=n_pred_steps,
            n_pred_features=n_features,
            seg_len=self.config.seg_len if seg_len is None else seg_len,
            d_model=self.config.d_model if d_model is None else d_model,
            dropout=self.config.dropout if dropout is None else dropout,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            optimizer=Adam(),
            device=self.config.device if device is None else device,
            saving_path=None,
        )

    @property
    def n_steps(self) -> int:
        """Return the fixed history length expected by SegRNN."""
        return (
            self._get_training_config_value("model", "n_steps")
            or self.config.n_steps
            or (_raise_missing_n_steps())
        )

    @property
    def n_pred_steps(self) -> int:
        """Return the forecast horizon expected by SegRNN."""
        return (
            self._get_training_config_value("model", "n_pred_steps")
            or self.config.n_pred_steps
            or self.forecasting_config.forecasting_length
        )

    @property
    def n_features(self) -> int:
        """Return the channel count expected by SegRNN."""
        return (
            self._get_training_config_value("model", "n_features")
            or self.config.n_features
            or infer_n_features(self.features_config)
        )


def _raise_missing_n_steps() -> int:
    raise ValueError(
        "Unable to infer evaluation history length for SegRNN. "
        "Set model.segrnn.n_steps in the eval config or use a checkpoint that includes "
        "training_config.json/training_config.yaml."
    )
