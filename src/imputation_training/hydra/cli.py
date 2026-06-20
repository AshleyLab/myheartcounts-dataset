"""Hydra entry point for ``mhc-impute-train``.

Mirrors :mod:`imputation_evaluation.hydra.cli` in structure: register
the :class:`PyPOTSTrainingConfig` dataclass tree with the Hydra
ConfigStore, resolve the absolute path to ``configs/training/`` (so the
console script works the same as ``python -m``), then delegate to
:func:`imputation_training.runner.run_training`.

Example::

    mhc-impute-train model=fedformer \
        seed=42 \
        data.version=full \
        data.daily_hf_dir=/path/to/daily_hf \
        +data.split_file=/path/to/sharable_users.json \
        training.epochs=50 \
        output.release_dir=/path/to/my-fedformer-release
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from eval_hydra import dict_to_dataclass, register_dataclass_tree
from imputation_evaluation.config import DataConfig
from imputation_training.config import (
    H5ExportConfig,
    ModelConfig,
    OutputConfig,
    PyPOTSTrainingConfig,
    TrainingConfig,
)
from imputation_training.runner import run_training

logger = logging.getLogger(__name__)


def register_configs() -> None:
    """Register the training-config dataclass tree with Hydra's ConfigStore.

    Called at module import time so the schema is available before
    ``@hydra.main`` composes the defaults list. Idempotent.
    """
    cs = ConfigStore.instance()
    register_dataclass_tree(
        cs,
        root_cls=PyPOTSTrainingConfig,
        root_name="imputation_train_schema",
        group_map={
            "data": DataConfig,
            "h5_export": H5ExportConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "output": OutputConfig,
        },
    )


register_configs()


# Resolve ``configs/training/`` to an absolute path at import time so the
# console-script entry point (``mhc-impute-train``) works the same as
# ``python -m imputation_training.hydra.cli``.
# Layout: this file lives at ``src/imputation_training/hydra/cli.py``;
# ``parents[3]`` is the repo root.
_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "training")


@hydra.main(
    version_base="1.3",
    config_path=_CONFIG_PATH,
    config_name="train",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run a Hydra-composed PyPOTS training job and return the release dir.

    Args:
        cfg: Hydra-composed config validated against
            :class:`PyPOTSTrainingConfig`.

    Returns:
        A dict with the staged ``release_dir`` path under the ``release_dir``
        key.
    """
    OmegaConf.resolve(cfg)
    typed_cfg: PyPOTSTrainingConfig = dict_to_dataclass(PyPOTSTrainingConfig, cfg)

    logger.info("Starting training: model=%s seed=%d", typed_cfg.model.model_name, typed_cfg.seed)
    release_dir = run_training(typed_cfg)
    logger.info("Done. Release bundle: %s", release_dir)
    return {"release_dir": str(release_dir)}


if __name__ == "__main__":
    main()
