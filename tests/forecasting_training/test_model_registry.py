"""The factory constructs each PyPOTS forecaster with accepted kwargs.

Guards the per-model constructor wiring — notably that MixLinear is built
WITHOUT an ``optimizer`` kwarg (PyPOTS MixLinear rejects it) while DLinear and
SegRNN pass ``Adam(lr=...)``. Construction only (no training), CPU.
"""

from __future__ import annotations

import pytest

from forecasting_training.config import ModelConfig, OutputConfig, TrainingConfig
from forecasting_training.model_registry import create_model


@pytest.mark.parametrize("name", ["dlinear", "mixlinear", "segrnn"])
def test_create_model_constructs(name: str, tmp_path) -> None:
    """create_model builds each supported forecaster with a fit method and inner module."""
    model_config = ModelConfig(
        model_name=name,
        n_steps=48,
        n_pred_steps=24,
        n_features=3,
        seg_len=24,  # divides n_steps and n_pred_steps for SegRNN
        period_len=24,  # MixLinear
    )
    # PyPOTS asserts patience < epochs at construction time.
    training_config = TrainingConfig(
        device="cpu", epochs=2, patience=1, batch_size=4, num_workers=0
    )
    output_config = OutputConfig(saving_path=str(tmp_path / "m"))

    model = create_model(model_config, training_config, output_config)
    assert model is not None
    assert hasattr(model, "fit")
    # PyPOTS NN models expose an inner nn.Module and (after construction) an optimizer.
    assert hasattr(model, "model")


def test_unknown_model_raises(tmp_path) -> None:
    """create_model raises ValueError for an unsupported model name."""
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model(
            ModelConfig(model_name="nope"),
            TrainingConfig(device="cpu"),
            OutputConfig(saving_path=str(tmp_path / "m")),
        )
