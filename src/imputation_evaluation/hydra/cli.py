"""Hydra entry point for ``mhc-impute-eval``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from eval_hydra import dict_to_dataclass, register_dataclass_tree, write_run_artifacts
from imputation_evaluation.config import (
    DataConfig,
    EvalConfig,
    ImputationEvalConfig,
    MaskingConfig,
    MethodConfig,
    OutputConfig,
    SensitivityConfig,
    VisualizationConfig,
    WandbConfig,
)
from imputation_evaluation.hydra.registry import METHOD_REGISTRY
from imputation_evaluation.runner import run_eval

logger = logging.getLogger(__name__)


def register_configs() -> None:
    """Register the imputation dataclass tree with Hydra's ConfigStore.

    Called at module import time so the schema is available before
    ``@hydra.main`` composes the defaults list. Safe to call repeatedly.
    """
    cs = ConfigStore.instance()
    register_dataclass_tree(
        cs,
        root_cls=ImputationEvalConfig,
        root_name="imputation_eval_schema",
        group_map={
            "data": DataConfig,
            "masking": MaskingConfig,
            "method": MethodConfig,
            "output": OutputConfig,
            "evaluation": EvalConfig,
            "visualization": VisualizationConfig,
            "sensitivity": SensitivityConfig,
            "wandb": WandbConfig,
        },
    )


register_configs()


# Resolve ``configs/imputation/`` to an absolute path at import time so the
# console-script entry point (``mhc-impute-eval``) works the same as
# ``python -m imputation_evaluation.hydra.cli``. Hydra's relative-path
# resolution gets confused by the entry-point wrapper, so we sidestep it.
# Layout: this file lives at ``src/imputation_evaluation/hydra/cli.py``;
# ``parents[3]`` is the repo root.
_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "imputation")


@hydra.main(
    version_base="1.3",
    config_path=_CONFIG_PATH,
    config_name="eval",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    OmegaConf.resolve(cfg)
    typed_cfg: ImputationEvalConfig = dict_to_dataclass(ImputationEvalConfig, cfg)

    method, manifest = METHOD_REGISTRY.build(cfg.method.type, typed_cfg.method, typed_cfg.data)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    wandb_run_id: str | None = None
    if typed_cfg.wandb.enabled:
        from imputation_evaluation.io.wandb_logger import finish, init_wandb, log_results

        init_wandb(typed_cfg)
        import wandb

        wandb_run_id = wandb.run.id if wandb.run is not None else None

    write_run_artifacts(
        run_dir,
        resolved_cfg=cfg,
        manifest=manifest,
        wandb_run_id=wandb_run_id,
    )

    logger.info("Running imputation eval (method=%s) → %s", cfg.method.type, run_dir)
    results = run_eval(typed_cfg, method=method)

    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_serializer)
    )

    if typed_cfg.wandb.enabled:
        log_results(results)
        finish()

    return results


def _json_serializer(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
