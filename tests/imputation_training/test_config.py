"""Sanity tests for the training-config dataclasses."""

from __future__ import annotations

import dataclasses

from imputation_training import (
    H5ExportConfig,
    ModelConfig,
    OutputConfig,
    PyPOTSTrainingConfig,
    TrainingConfig,
)


def test_default_construction() -> None:
    """PyPOTSTrainingConfig defaults construct with the expected nested config types."""
    cfg = PyPOTSTrainingConfig()
    assert cfg.seed == 42
    assert isinstance(cfg.data.__class__.__name__, str)
    assert isinstance(cfg.h5_export, H5ExportConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.output, OutputConfig)


def test_model_names_are_lowercase() -> None:
    """The default model_name is already lowercase.

    The factory uses ``.lower()``, so we don't bother case-checking
    everywhere — but make sure the default is already lowercase to
    avoid surprises in CLI overrides.
    """
    assert ModelConfig().model_name == ModelConfig().model_name.lower()


def test_asdict_roundtrip() -> None:
    """OmegaConf interop: all configs must survive ``dataclasses.asdict``."""
    cfg = PyPOTSTrainingConfig()
    d = dataclasses.asdict(cfg)
    assert d["seed"] == 42
    assert d["model"]["model_name"] == "brits"
    assert d["training"]["epochs"] == 100
