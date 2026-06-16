"""Run-config loading and copying helpers for offline metrics."""

from __future__ import annotations

from dataclasses import fields
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


def _known_fields(config_cls: type, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only keys that are accepted dataclass fields of ``config_cls``.

    Run configs are persisted alongside predictions for post-hoc metric
    recomputation, so a config written by an older schema (e.g. carrying a since-
    removed field like ``seasonal_naive_average_history``) must still load. Unknown
    top-level keys are dropped rather than raising ``TypeError``.
    """
    allowed = {f.name for f in fields(config_cls)}
    return {key: value for key, value in (raw or {}).items() if key in allowed}


def load_run_config(run_path: Path) -> ForecastingEvalConfig:
    """Load forecasting config from run directory config.yaml."""
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml under run path: {run_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg: dict[str, Any] = yaml.safe_load(handle)

    data_cfg = DataConfig(**_known_fields(DataConfig, raw_cfg.get("data", {})))

    forecasting_raw = raw_cfg.get("forecasting", {})
    forecasting_cfg = ForecastingConfig(
        forecasting_length=forecasting_raw.get("forecasting_length", 24),
        daily_start_hour_offset=forecasting_raw.get("daily_start_hour_offset", 0),
    )

    model_cfg = ForecastingModelConfig(**_known_fields(ForecastingModelConfig, raw_cfg.get("model", {})))
    features_cfg = FeaturesConfig(**_known_fields(FeaturesConfig, raw_cfg.get("features", {})))
    output_cfg = OutputConfig(**_known_fields(OutputConfig, raw_cfg.get("output", {})))

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
