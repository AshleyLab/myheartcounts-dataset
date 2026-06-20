"""Results writing utilities for imputation evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from imputation_evaluation.config import ImputationEvalConfig, OutputConfig

logger = logging.getLogger(__name__)


def resolve_experiment_name(config: OutputConfig, full_config: ImputationEvalConfig) -> str:
    """Resolve the final experiment name from output config.

    Args:
        config: Output configuration.
        full_config: Full imputation evaluation config.

    Returns:
        Final experiment name, including optional prefix.
    """
    if config.experiment_name:
        base_experiment_name = config.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_experiment_name = f"imputation_{full_config.method.type}_{timestamp}"

    if config.experiment_name_prefix:
        return f"{config.experiment_name_prefix}_{base_experiment_name}"

    return base_experiment_name


class ResultsWriter:
    """Write evaluation results to JSON/YAML files."""

    def __init__(self, config: OutputConfig, full_config: ImputationEvalConfig):
        """Initialize results writer.

        Args:
            config: Output configuration.
            full_config: Full imputation evaluation config for saving.
        """
        self.config = config
        self.full_config = full_config

        self.experiment_name = resolve_experiment_name(config, full_config)
        self.output_dir = Path(config.results_dir) / self.experiment_name

    def write(self, results: dict) -> Path:
        """Write results to output directory.

        Creates:
        - results.json: Full results with per-scenario metrics
        - config.yaml: Copy of config used

        Args:
            results: Results dictionary from evaluator.

        Returns:
            Path to output directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write main results JSON
        results_file = self.output_dir / "results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)
        logger.info(f"Saved results to {results_file}")

        # Write config
        if self.config.save_config:
            config_file = self.output_dir / "config.yaml"
            self.full_config.output.experiment_name = self.experiment_name
            self.full_config.output.experiment_name_prefix = None
            config_dict = asdict(self.full_config)
            with config_file.open("w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {config_file}")

        logger.info(f"Results saved to: {self.output_dir}")
        return self.output_dir


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    import numpy as np

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
