"""Config defaults + asdict round-trip for ForecastingTrainingConfig."""

from __future__ import annotations

from dataclasses import asdict

from forecasting_training.config import ForecastingTrainingConfig


def test_defaults() -> None:
    """ForecastingTrainingConfig defaults match the eval-consistent training setup."""
    cfg = ForecastingTrainingConfig()
    assert cfg.model.model_name == "dlinear"
    assert cfg.model.n_steps == 168
    assert cfg.model.n_pred_steps == 24
    assert cfg.model.n_features == 19
    # Eval-consistent training defaults.
    assert cfg.training.whether_standardscaler is True
    assert cfg.training.include_short_history is True
    assert cfg.output.wandb_project == "mhc-forecasting"


def test_asdict_sections() -> None:
    """Serializing via asdict exposes the config sections the eval adapter reads."""
    d = asdict(ForecastingTrainingConfig())
    # The eval adapter reads these sections from the written training_config.json.
    assert set(d) >= {"data", "forecasting", "features", "model", "training", "output"}
    assert d["training"]["whether_standardscaler"] is True
    assert d["forecasting"]["daily_start_hour_offset"] == 0


def test_offset_suffixes_saving_path() -> None:
    """__post_init__ suffixes saving_path with a nonzero daily_start_hour_offset only."""
    cfg = ForecastingTrainingConfig()
    cfg.forecasting.daily_start_hour_offset = 0
    cfg.output.saving_path = "models/forecasting_pypots"
    ForecastingTrainingConfig.__post_init__(cfg)  # offset 0 -> no-op
    assert cfg.output.saving_path == "models/forecasting_pypots"

    cfg.forecasting.daily_start_hour_offset = 6
    cfg.output.saving_path = "models/forecasting_pypots"
    ForecastingTrainingConfig.__post_init__(cfg)
    assert cfg.output.saving_path == "models/forecasting_pypots_6"
