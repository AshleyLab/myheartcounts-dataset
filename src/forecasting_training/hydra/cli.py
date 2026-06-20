"""Hydra entry point for ``mhc-forecast-train``.

Mirrors :mod:`imputation_training.hydra.cli` and
:mod:`forecasting_evaluation.hydra.cli`: register the
:class:`ForecastingTrainingConfig` dataclass tree with the Hydra ConfigStore,
resolve the absolute path to ``configs/forecasting_train/`` (so the console
script behaves like ``python -m``), then delegate to
:func:`forecasting_training.runner.run_training`.

Example::

    mhc-forecast-train model=dlinear \
        seed=42 \
        data.trajectory_hf_dir=/path/to/hourly_trajectory \
        +data.split_file=/path/to/sharable_users.json \
        data.sample_index_file=/path/to/sample_index.json \
        training.epochs=50 \
        output.release_dir=/path/to/my-dlinear-release
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from eval_hydra import dict_to_dataclass, register_dataclass_tree
from forecasting_evaluation.config import (
    DataConfig,
    FeaturesConfig,
    ForecastingConfig,
)
from forecasting_training.config import (
    ForecastingTrainingConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from forecasting_training.runner import run_training

logger = logging.getLogger(__name__)


def register_configs() -> None:
    """Register the training-config dataclass tree with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    register_dataclass_tree(
        cs,
        root_cls=ForecastingTrainingConfig,
        root_name="forecasting_train_schema",
        group_map={
            "data": DataConfig,
            "forecasting": ForecastingConfig,
            "features": FeaturesConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "output": OutputConfig,
        },
    )


register_configs()


# Resolve ``configs/forecasting_train/`` to an absolute path at import time so the
# console-script entry point works the same as ``python -m``.
# Layout: this file lives at ``src/forecasting_training/hydra/cli.py``;
# ``parents[3]`` is the repo root.
_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "forecasting_train")


@hydra.main(
    version_base="1.3",
    config_path=_CONFIG_PATH,
    config_name="train",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Compose the config, build the typed config, and run training."""
    OmegaConf.resolve(cfg)
    typed_cfg: ForecastingTrainingConfig = dict_to_dataclass(ForecastingTrainingConfig, cfg)

    logger.info(
        "Starting forecasting training: model=%s seed=%d",
        typed_cfg.model.model_name,
        typed_cfg.seed,
    )
    release_dir = run_training(typed_cfg)
    logger.info("Done. Release bundle: %s", release_dir)
    return {"release_dir": str(release_dir)}


if __name__ == "__main__":
    main()
