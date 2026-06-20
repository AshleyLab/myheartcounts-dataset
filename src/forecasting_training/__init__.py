"""PyPOTS forecasting-model training pipeline for OpenMHC.

The companion to :mod:`forecasting_evaluation`: same data layer, same splits and
sample index, and produces release bundles directly consumable by the eval
Hydra CLI's ``model.release_dir=...`` flag.

Supported models (the three trainable PyPOTS forecasters benchmarked in the
OpenMHC paper):
- DLinear
- MixLinear
- SegRNN

Training keeps short-history windows (NaN-left-padded to the model's fixed
``n_steps``) by default, matching the distribution the evaluator feeds at
inference time — so trained and evaluated input distributions agree.

Public API:
    >>> from forecasting_training import ForecastingTrainingConfig, run_training
    >>> cfg = ForecastingTrainingConfig(...)
    >>> release_dir = run_training(cfg)
"""

from __future__ import annotations

from forecasting_training.config import (
    ForecastingTrainingConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from forecasting_training.runner import run_training
from forecasting_training.seeding import seed_everything

__all__ = [
    "ForecastingTrainingConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "run_training",
    "seed_everything",
]
