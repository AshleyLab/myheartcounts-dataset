"""Hydra entry point for ``mhc-forecast-eval``."""

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
from forecasting_evaluation.config import (
    DataConfig,
    EvaluatorConfig,
    FeaturesConfig,
    ForecastingConfig,
    ForecastingEvalConfig,
    ForecastingModelConfig,
    OutputConfig,
)
from forecasting_evaluation.hydra.registry import MODEL_REGISTRY
from forecasting_evaluation.runner import run_eval

logger = logging.getLogger(__name__)


def register_configs() -> None:
    cs = ConfigStore.instance()
    register_dataclass_tree(
        cs,
        root_cls=ForecastingEvalConfig,
        root_name="forecasting_eval_schema",
        group_map={
            "data": DataConfig,
            "forecasting": ForecastingConfig,
            "model": ForecastingModelConfig,
            "features": FeaturesConfig,
            "evaluator": EvaluatorConfig,
            "output": OutputConfig,
        },
    )


register_configs()


# Resolve ``configs/forecasting/`` to an absolute path at import time so the
# console-script entry point (``mhc-forecast-eval``) works the same as
# ``python -m forecasting_evaluation.hydra.cli``. Hydra's relative-path
# resolution gets confused by the entry-point wrapper, so we sidestep it.
# Layout: this file lives at ``src/forecasting_evaluation/hydra/cli.py``;
# ``parents[3]`` is the repo root.
_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "forecasting")


@hydra.main(
    version_base="1.3",
    config_path=_CONFIG_PATH,
    config_name="eval",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    OmegaConf.resolve(cfg)
    typed_cfg: ForecastingEvalConfig = dict_to_dataclass(ForecastingEvalConfig, cfg)

    model, manifest = MODEL_REGISTRY.build(
        cfg.model.type,
        typed_cfg.model,
        typed_cfg.forecasting,
        typed_cfg.features,
        typed_cfg.seed,
    )

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    write_run_artifacts(run_dir, resolved_cfg=cfg, manifest=manifest)

    logger.info("Running forecasting eval (model=%s) → %s", cfg.model.type, run_dir)
    results = run_eval(typed_cfg, model=model)

    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_serializer)
    )

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
