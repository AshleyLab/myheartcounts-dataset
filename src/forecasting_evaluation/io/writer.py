"""Results writing utilities for downstream evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ResultsWriter:
    """Write evaluation results to JSON/CSV files."""

    def __init__(self, config, full_config):
        """Initialize results writer.

        Args:
            config: Output configuration.
            full_config: Full downstream evaluation config for saving.
        """
        self.config = config
        self.full_config = full_config

        # Generate experiment name if not provided
        if config.experiment_name:
            self.experiment_name = config.experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"{full_config.data.task_name}_"
                f"{full_config.features.type}_"
                f"{full_config.classifier.type}_"
                f"{timestamp}"
            )

        self.output_dir = Path(config.results_dir) / self.experiment_name

    def write(self, results: dict) -> Path:
        """Write results to output directory.

        Creates:
        - results.json: Full results with metrics
        - config.yaml: Copy of config used
        - predictions_val.csv: Per-user val predictions (if save_predictions=True)
        - predictions_test.csv: Per-user test predictions (if save_predictions=True)

        Args:
            results: Results dictionary from evaluator.

        Returns:
            Path to output directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Separate predictions for CSV output
        predictions = results.pop("predictions", None)

        # Write main results JSON
        results_file = self.output_dir / "results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_file}")

        # Write config
        if self.config.save_config:
            config_file = self.output_dir / "config.yaml"
            config_dict = asdict(self.full_config)
            with config_file.open("w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {config_file}")

        # Write per-user predictions as CSV
        if predictions and self.config.save_predictions:
            try:
                import pandas as pd

                for split_name, pred_list in predictions.items():
                    csv_file = self.output_dir / f"predictions_{split_name}.csv"
                    pd.DataFrame(pred_list).to_csv(csv_file, index=False)
                    logger.info(f"Saved predictions to {csv_file}")
            except ImportError:
                logger.warning("pandas not available, skipping CSV prediction output")

        logger.info(f"Results saved to: {self.output_dir}")
        return self.output_dir
