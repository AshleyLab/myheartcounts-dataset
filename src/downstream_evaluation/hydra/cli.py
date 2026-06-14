"""Hydra entry point for ``mhc-downstream-eval``.

The downstream (health-prediction) twin of ``mhc-impute-eval``: compose a
:class:`~downstream_evaluation.config.DownstreamEvalConfig` from
``configs/downstream/*``, build the bundled model from the registry, run it
through the public ``openmhc.evaluate_prediction`` API, and write the eval CSV
+ resolved config into the Hydra run directory.

Replaces the env-var-driven ``scripts/run_eval.py`` (``METHOD=…``,
``MHC_DATA_DIR=…``, ``PREDICTIONS_DIR=…``, ``MAE_CHECKPOINT=…``) with composable
configs:

    mhc-downstream-eval method=xgboost
    mhc-downstream-eval --multirun method=linear,mae,xgboost
    mhc-downstream-eval method=mae data.data_dir=/path output.predictions_dir=preds/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from downstream_evaluation.config import (
    DataConfig,
    DownstreamEvalConfig,
    EvaluationConfig,
    MethodConfig,
    OutputConfig,
)
from downstream_evaluation.hydra.registry import METHOD_REGISTRY
from eval_hydra import dict_to_dataclass, register_dataclass_tree, write_run_artifacts

logger = logging.getLogger(__name__)


def register_configs() -> None:
    """Register the downstream dataclass tree with Hydra's ConfigStore.

    Called at import time so the schema is available before ``@hydra.main``
    composes the defaults list. Safe to call repeatedly.
    """
    cs = ConfigStore.instance()
    register_dataclass_tree(
        cs,
        root_cls=DownstreamEvalConfig,
        root_name="downstream_eval_schema",
        group_map={
            "data": DataConfig,
            "method": MethodConfig,
            "evaluation": EvaluationConfig,
            "output": OutputConfig,
        },
    )


register_configs()


# Resolve ``configs/downstream/`` to an absolute path at import time so the
# console-script entry point works the same as ``python -m
# downstream_evaluation.hydra.cli``. This file lives at
# ``src/downstream_evaluation/hydra/cli.py``; ``parents[3]`` is the repo root.
_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "downstream")


@hydra.main(version_base="1.3", config_path=_CONFIG_PATH, config_name="eval")
def main(cfg: DictConfig) -> Any:
    """Compose the config, build the model, run ``evaluate_prediction``, write artifacts."""
    OmegaConf.resolve(cfg)
    typed_cfg: DownstreamEvalConfig = dict_to_dataclass(DownstreamEvalConfig, cfg)

    model, manifest = METHOD_REGISTRY.build(cfg.method.type, typed_cfg.method, typed_cfg.data)

    import openmhc

    results = openmhc.evaluate_prediction(
        model,
        tasks=typed_cfg.evaluation.tasks,
        data_dir=typed_cfg.data.data_dir,
        seed=typed_cfg.seed,
        predictions_dir=typed_cfg.output.predictions_dir,
    )

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    results.to_csv(run_dir / "eval.csv")
    write_run_artifacts(run_dir, resolved_cfg=cfg, manifest=manifest)
    logger.info("downstream eval (%s) done -> %s", cfg.method.type, run_dir / "eval.csv")
    return results.records


if __name__ == "__main__":
    main()
