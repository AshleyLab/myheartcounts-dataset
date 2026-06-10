"""Thin orchestrator for structured offline forecasting metrics pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forecasting_evaluation.metrics.offline.pipeline import OfflineMetricsPipeline

logger = logging.getLogger(__name__)


class OfflineMetricsCalculator:
    """Run the structured offline metrics pipeline for one or more forecasting runs."""

    def __init__(
        self,
        evaluation_result_paths: dict[str, str],
        metrics_output_path: str,
        max_user: int | None = None,
        combine_channels: bool = True,
        metric_columns: tuple[str, ...] | None = None,
    ):
        """Initialize the multi-run offline metrics calculator."""
        self.evaluation_result_paths = {
            str(name): Path(path) for name, path in evaluation_result_paths.items()
        }
        self.metrics_output_path = Path(metrics_output_path)
        self.metrics_output_path.mkdir(parents=True, exist_ok=True)
        self.max_user = int(max_user) if max_user is not None else None
        self.combine_channels = bool(combine_channels)
        self.metric_columns = tuple(metric_columns) if metric_columns else None

    def run(self) -> dict[str, Any]:
        """Run the offline metrics pipeline for all provided runs."""
        run_summaries: list[dict[str, Any]] = []

        for run_key, run_path in self.evaluation_result_paths.items():
            if not run_path.exists():
                logger.warning("Run path does not exist for key=%s, skipped: %s", run_key, run_path)
                continue

            logger.info("Processing key=%s run path=%s", run_key, run_path)
            pipeline = OfflineMetricsPipeline(
                run_key=run_key,
                run_path=run_path,
                metrics_output_path=self.metrics_output_path,
                max_user=self.max_user,
                combine_channels=self.combine_channels,
                metric_columns=self.metric_columns,
            )
            run_summaries.append(pipeline.run())

        total_saved = int(sum(summary["saved_rows"] for summary in run_summaries))
        total_skipped = int(sum(summary["skipped_rows"] for summary in run_summaries))

        return {
            "runs": run_summaries,
            "total_runs": len(run_summaries),
            "total_saved_rows": total_saved,
            "total_skipped_rows": total_skipped,
        }
