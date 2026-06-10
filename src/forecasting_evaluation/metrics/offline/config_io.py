"""Run-config loading and copying helpers for offline metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from forecasting_evaluation.config import (
    DataConfig,
    FeaturesConfig,
    ForecastingConfig,
    ForecastingEvalConfig,
    ForecastingModelConfig,
    OutputConfig,
)


def load_run_config(run_path: Path) -> ForecastingEvalConfig:
    """Load forecasting config from run directory config.yaml."""
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml under run path: {run_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg: dict[str, Any] = yaml.safe_load(handle)

    data_cfg = DataConfig(**raw_cfg.get("data", {}))

    forecasting_raw = raw_cfg.get("forecasting", {})
    forecasting_cfg = ForecastingConfig(
        forecasting_length=forecasting_raw.get("forecasting_length", 24),
        daily_start_hour_offset=forecasting_raw.get("daily_start_hour_offset", 0),
    )

    model_cfg = ForecastingModelConfig(**raw_cfg.get("model", {}))
    features_cfg = FeaturesConfig(**raw_cfg.get("features", {}))
    output_cfg = OutputConfig(**raw_cfg.get("output", {}))

    return ForecastingEvalConfig(
        seed=raw_cfg.get("seed", 42),
        experiment_name=raw_cfg.get("experiment_name"),
        debug_mode=raw_cfg.get("debug_mode", False),
        time_granularity=raw_cfg.get("time_granularity", "hourly"),
        data=data_cfg,
        forecasting=forecasting_cfg,
        model=model_cfg,
        features=features_cfg,
        output=output_cfg,
    )


def copy_run_config(source_run_dir: Path, target_run_dir: Path) -> None:
    """Copy run config.yaml into metrics output directory for reproducibility."""
    src = source_run_dir / "config.yaml"
    dst = target_run_dir / "config.yaml"
    if not src.exists() or dst.exists():
        return
    target_run_dir.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
