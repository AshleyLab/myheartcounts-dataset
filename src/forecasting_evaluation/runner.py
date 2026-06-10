"""Library entry-point for the forecasting eval pipeline.

The script ``scripts/run_forecasting_eval.py`` in the private repo runs a
2-phase pipeline: ``ForecastingEvaluator.run()`` writes per-user prediction
parquets, then ``OfflineMetricsCalculator.run()`` aggregates them into
per-channel metrics. This module exposes a single ``run_eval(config,
model)`` function that does both steps and returns the aggregated metrics
in-memory so the public OpenMHC API can call it without orchestrating the
full script.

A custom forecaster is supplied via the ``model`` argument (any object
satisfying :class:`forecasting_evaluation.models.base.BasePredictionModel`)
so the registry's built-in models aren't required. The internal evaluator
constructs its own model from the config — we monkey-patch that step by
overriding ``run()`` on a thin subclass.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from forecasting_evaluation.evaluation.evaluator import ForecastingEvaluator
from forecasting_evaluation.io.predict_result_writer import PublicWriter

if TYPE_CHECKING:
    from forecasting_evaluation.config import ForecastingEvalConfig
    from forecasting_evaluation.models.base import BasePredictionModel

logger = logging.getLogger(__name__)


class _CustomModelEvaluator(ForecastingEvaluator):
    """Subclass that injects a pre-constructed model rather than using the registry."""

    def __init__(self, config: "ForecastingEvalConfig", model: "BasePredictionModel"):
        super().__init__(config)
        self._injected_model = model

    def run(self) -> dict:
        """Same flow as the parent class but skips ``create_forecasting_model``."""
        from forecasting_evaluation.config import print_config

        print_config(self.config)
        model = self._injected_model
        data_context = self._load_evaluation_data()

        public_writer = PublicWriter(
            self.config,
            experiment_name=self.config.experiment_name,
        )
        self._run_sequential(model, data_context, public_writer)
        run_dir = public_writer.finalize()
        return {
            "run_dir": str(run_dir),
            "prediction_samples": public_writer.total_written,
            "skipped_users": public_writer.skipped_users,
            **self._fallback_summary,
        }


def run_eval(
    config: "ForecastingEvalConfig",
    model: "BasePredictionModel",
    metrics_output_dir: str | Path | None = None,
) -> dict:
    """Run forecasting eval + offline metrics in one call.

    Args:
        config: Fully-populated :class:`ForecastingEvalConfig`.
        model: Pre-constructed model (any object satisfying the
            :class:`BasePredictionModel` protocol — has ``predict()``).
        metrics_output_dir: Where the offline-metrics pipeline writes its
            outputs. Defaults to a temp directory adjacent to the eval
            run dir.

    Returns:
        Dict with keys:
        - ``run_dir``: where prediction parquets were written
        - ``metrics_dir``: where offline metrics were written
        - ``per_channel``: aggregated per-channel metrics
        - ``n_samples``: prediction samples emitted
        - ``overall_fallback_rate``: fraction of forecast cells where the model
          returned NaN and the Seasonal-Naive baseline was substituted
        - ``fallback_rate``: per-channel Seasonal-Naive substitution fractions
    """
    # Phase 1: prediction generation.
    evaluator = _CustomModelEvaluator(config, model)
    eval_summary = evaluator.run()
    run_dir = Path(eval_summary["run_dir"])

    # Phase 2: offline metric aggregation.
    from forecasting_evaluation.metrics.offline.runner import OfflineMetricsCalculator

    if metrics_output_dir is None:
        metrics_output_dir = run_dir.parent / f"{run_dir.name}_metrics"
    metrics_output_dir = Path(metrics_output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_cfg = getattr(config, "metrics", None)
    run_key = config.experiment_name or "openmhc_run"
    eval_paths = {run_key: str(run_dir)}
    point_metrics = tuple(metrics_cfg.point_metrics) if metrics_cfg else None
    combine_channels = bool(metrics_cfg.combine_channels) if metrics_cfg else True

    calculator = OfflineMetricsCalculator(
        evaluation_result_paths=eval_paths,
        metrics_output_path=str(metrics_output_dir),
        combine_channels=combine_channels,
        metric_columns=point_metrics,
    )
    calculator.run()

    # Binary-channel metrics (f1/auroc/auprc) into the SAME metrics tree (same
    # run_key), so the skill/ranking scripts read one model-root containing both
    # point and binary metrics. Skipped when binary_metrics is empty.
    binary_metrics = tuple(metrics_cfg.binary_metrics) if metrics_cfg else ()
    if binary_metrics:
        from forecasting_evaluation.metrics.binary_offline_calculate import (
            BinaryOfflineMetricsCalculator,
        )

        BinaryOfflineMetricsCalculator(
            evaluation_result_paths=eval_paths,
            metrics_output_path=str(metrics_output_dir),
            threshold=float(metrics_cfg.f1_threshold),
            combine_channels=combine_channels,
            metric_names=binary_metrics,
        ).run()

    per_channel = _load_per_channel_metrics(metrics_output_dir)

    return {
        "run_dir": str(run_dir),
        "metrics_dir": str(metrics_output_dir),
        "per_channel": per_channel,
        "n_samples": int(eval_summary["prediction_samples"]),
        "overall_fallback_rate": float(eval_summary.get("overall_fallback_rate", 0.0)),
        "fallback_rate": dict(eval_summary.get("fallback_rate", {})),
    }


def _load_per_channel_metrics(metrics_dir: Path) -> dict[str, dict]:
    """Aggregate per-channel metrics from offline-pipeline outputs.

    Offline-pipeline layout (set by
    :func:`metrics.offline.parquet_io.save_metrics_result_by_metric`)::

        <metrics_dir>/<run_key>/<metric>/<user_id>.parquet

    Each parquet has rows ``(user_id, model, history_length,
    forecasting_length, <metric>)``. The ``<metric>`` cell holds a 2D
    nested-list of shape ``(n_features, horizon)`` (i.e. MAE per channel
    per horizon timestep). We average across rows, users, and horizon to
    get one scalar per (channel, metric) and key the output by channel
    index ``ch_<i>``.
    """
    import numpy as np
    import pandas as pd

    out: dict[str, dict] = {}
    metric_dirs = [p for p in metrics_dir.rglob("*") if p.is_dir() and any(p.glob("*.parquet"))]
    for metric_dir in metric_dirs:
        metric_name = metric_dir.name
        # Stack all per-user values into one (n_rows, n_features, horizon) array.
        per_channel_values: dict[int, list[float]] = {}
        for parquet_path in metric_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
            if metric_name not in df.columns:
                continue
            for cell in df[metric_name]:
                if cell is None:
                    continue
                # The parquet cell is a list-of-lists (n_features × horizon).
                # pandas may load it as object array of arrays; flatten per channel.
                try:
                    rows = list(cell)
                except TypeError:
                    continue
                for ch_idx, ch_row in enumerate(rows):
                    try:
                        ch_arr = np.asarray(list(ch_row), dtype=float)
                    except Exception:
                        continue
                    with np.errstate(invalid="ignore"):
                        ch_mean = float(np.nanmean(ch_arr)) if ch_arr.size else float("nan")
                    if np.isfinite(ch_mean):
                        per_channel_values.setdefault(ch_idx, []).append(ch_mean)
        for ch_idx, vals in per_channel_values.items():
            if not vals:
                continue
            channel_key = f"ch_{ch_idx}"
            out.setdefault(channel_key, {})[metric_name] = float(np.mean(vals))
    return out
