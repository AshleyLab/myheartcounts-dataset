"""CLI entry point for binary forecasting metrics calculation.

This companion to ``offline_calculate.py`` computes per-sample binary-channel
metrics from saved forecasting prediction parquets and stores them under the
same metric-partitioned output tree:

``results/metrics/<model_name>/{f1,auroc,auprc}/*.parquet``
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics.offline.channel_merge import (
    merge_channel_first_array,
    resolve_channel_merges,
)
from forecasting_evaluation.metrics.offline.common import (
    coerce_2d_float_array,
    coerce_non_negative_int,
    get_model_name,
    sanitize_name,
)
from forecasting_evaluation.metrics.offline.config_io import copy_run_config, load_run_config
from forecasting_evaluation.metrics.offline.data_context import (
    load_offline_user_contexts_from_eval_flow,
)
from forecasting_evaluation.metrics.offline.metric_core import slice_ground_truth
from forecasting_evaluation.metrics.offline.parquet_io import (
    index_prediction_files,
    read_parquet_rows,
)

logger = logging.getLogger(__name__)

_METRIC_NAMES = ("f1", "auroc", "auprc")


def _parse_named_paths(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid run mapping '{item}', expected format: name=/abs/or/rel/path"
            )
        key, path = item.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise argparse.ArgumentTypeError(
                f"Invalid run mapping '{item}', expected non-empty name and path"
            )
        parsed[key] = path
    return parsed


def _append_records_to_parquet(output_dir: Path, user_id: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sanitize_name(user_id)}.parquet"
    df_new = pd.DataFrame(records)
    if output_file.exists():
        df_existing = pd.read_parquet(output_file)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(output_file, engine="pyarrow", compression="snappy")


def _save_binary_metrics_by_metric(
    *,
    output_root: Path,
    model_key: str,
    records_by_user: dict[str, list[dict[str, Any]]],
    metric_names: tuple[str, ...] = _METRIC_NAMES,
) -> dict[str, dict[str, str]]:
    saved: dict[str, dict[str, str]] = {}
    for metric_name in metric_names:
        metric_dir = output_root / metric_name
        metric_saved: dict[str, str] = {}
        for user_id, records in records_by_user.items():
            metric_rows: list[dict[str, Any]] = []
            for record in records:
                metric_rows.append(
                    {
                        "user_id": record.get("user_id"),
                        "model": record.get("model"),
                        "history_length": record.get("history_length"),
                        "forecasting_length": record.get("forecasting_length"),
                        metric_name: record.get(metric_name),
                        "binary_valid_count": record.get("binary_valid_count"),
                        "binary_positive_count": record.get("binary_positive_count"),
                        "binary_negative_count": record.get("binary_negative_count"),
                    }
                )
            if metric_rows:
                _append_records_to_parquet(metric_dir, user_id, metric_rows)
                metric_saved[user_id] = str(metric_dir / f"{sanitize_name(user_id)}.parquet")
        if metric_saved:
            saved[metric_name] = metric_saved
            logger.info(
                "Saved binary metrics for model=%s metric=%s users=%d dir=%s",
                model_key,
                metric_name,
                len(metric_saved),
                metric_dir,
            )
    return saved


def _compute_f1_from_scores(truth: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred_binary = score >= float(threshold)
    truth_binary = truth == 1.0
    tp = int(np.sum(truth_binary & pred_binary))
    fp = int(np.sum((~truth_binary) & pred_binary))
    fn = int(np.sum(truth_binary & (~pred_binary)))
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return float("nan")
    return float((2.0 * tp) / denominator)


def _compute_binary_metrics_for_sample(
    *,
    point_predictions: np.ndarray | None,
    ground_truth: np.ndarray,
    threshold: float,
) -> dict[str, list[float]]:
    n_features, _ = ground_truth.shape
    f1 = np.full(n_features, np.nan, dtype=float)
    auroc = np.full(n_features, np.nan, dtype=float)
    auprc = np.full(n_features, np.nan, dtype=float)
    valid_count = np.zeros(n_features, dtype=int)
    positive_count = np.zeros(n_features, dtype=int)
    negative_count = np.zeros(n_features, dtype=int)

    if point_predictions is None or point_predictions.shape != ground_truth.shape:
        return {
            "f1": f1.tolist(),
            "auroc": auroc.tolist(),
            "auprc": auprc.tolist(),
            "binary_valid_count": valid_count.tolist(),
            "binary_positive_count": positive_count.tolist(),
            "binary_negative_count": negative_count.tolist(),
        }

    for feature_idx in range(n_features):
        truth_row = np.asarray(ground_truth[feature_idx], dtype=float).reshape(-1)
        score_row = np.asarray(point_predictions[feature_idx], dtype=float).reshape(-1)
        finite_mask = np.isfinite(truth_row) & np.isfinite(score_row)
        if not np.any(finite_mask):
            continue

        truth_values = truth_row[finite_mask]
        score_values = score_row[finite_mask]
        positive_mask = truth_values == 1.0
        negative_mask = truth_values == 0.0
        binary_mask = positive_mask | negative_mask
        if not np.any(binary_mask):
            continue

        truth_binary = truth_values[binary_mask]
        score_binary = score_values[binary_mask]
        n_positive = int(np.sum(truth_binary == 1.0))
        n_negative = int(np.sum(truth_binary == 0.0))
        positive_count[feature_idx] = n_positive
        negative_count[feature_idx] = n_negative
        valid_count[feature_idx] = int(truth_binary.shape[0])

        if n_positive <= 0:
            continue

        f1[feature_idx] = _compute_f1_from_scores(
            truth=truth_binary,
            score=score_binary,
            threshold=threshold,
        )
        auprc[feature_idx] = float(average_precision_score(truth_binary, score_binary))
        if n_negative > 0:
            auroc[feature_idx] = float(roc_auc_score(truth_binary, score_binary))

    return {
        "f1": f1.tolist(),
        "auroc": auroc.tolist(),
        "auprc": auprc.tolist(),
        "binary_valid_count": valid_count.tolist(),
        "binary_positive_count": positive_count.tolist(),
        "binary_negative_count": negative_count.tolist(),
    }


class BinaryOfflineMetricsPipeline:
    """Compute and persist per-sample binary metrics for one forecasting run."""

    def __init__(
        self,
        *,
        run_key: str,
        run_path: Path,
        metrics_output_path: Path,
        threshold: float,
        max_user: int | None = None,
        combine_channels: bool = True,
        metric_names: tuple[str, ...] = _METRIC_NAMES,
    ):
        """Initialize binary metric generation for one forecasting run."""
        self.run_key = str(run_key)
        self.run_path = Path(run_path)
        self.metrics_output_path = Path(metrics_output_path)
        self.threshold = float(threshold)
        self.max_user = int(max_user) if max_user is not None else None
        self.combine_channels = bool(combine_channels)
        self.metric_names = tuple(metric_names)

    def run(self) -> dict[str, Any]:
        """Compute and save binary metrics for one forecasting run."""
        config = load_run_config(self.run_path)
        forecast_length = int(config.forecasting.forecasting_length)
        model_name = get_model_name(config.model)
        prediction_files = index_prediction_files(self.run_path, model_name)

        output_run_dir = self.metrics_output_path / sanitize_name(self.run_key)
        copy_run_config(self.run_path, output_run_dir)

        user_contexts = load_offline_user_contexts_from_eval_flow(
            config=config,
            prediction_files=prediction_files,
        )
        for user_context in user_contexts.values():
            history = np.asarray(user_context["history"], dtype=float)
            variable_names = list(user_context["variable_names"])
            merge_plan = resolve_channel_merges(
                variable_names, combine_channels=self.combine_channels
            )
            user_context["history"] = merge_channel_first_array(history, merge_plan)
            user_context["merge_plan"] = merge_plan

        prediction_rows_by_user: dict[str, list[dict[str, Any]]] = {}
        for user_id, parquet_paths in prediction_files.items():
            rows: list[dict[str, Any]] = []
            for parquet_path in parquet_paths:
                rows.extend(read_parquet_rows(parquet_path))
            if rows:
                prediction_rows_by_user[user_id] = rows

        records_by_user: dict[str, list[dict[str, Any]]] = {}
        saved_rows = 0
        skipped_rows = 0
        computed_user_count = 0

        for user_id, user_context in user_contexts.items():
            if self.max_user is not None and computed_user_count >= self.max_user:
                logger.info(
                    "Reached max_user limit for binary metrics run=%s max_user=%s",
                    self.run_key,
                    self.max_user,
                )
                break

            user_metrics_file = output_run_dir / "f1" / f"{sanitize_name(user_id)}.parquet"
            if user_metrics_file.exists():
                logger.info(
                    "Binary metrics parquet for user exists, skipping. run=%s user=%s file=%s",
                    self.run_key,
                    user_id,
                    user_metrics_file,
                )
                continue

            history = np.asarray(user_context["history"], dtype=float)
            merge_plan = user_context["merge_plan"]
            user_records: list[dict[str, Any]] = []
            for row in prediction_rows_by_user.get(user_id, []):
                history_length = coerce_non_negative_int(row.get("history_length"))
                if history_length is None:
                    skipped_rows += 1
                    continue

                gt_pack = slice_ground_truth(
                    history=history,
                    history_length=history_length,
                    forecast_length=forecast_length,
                )
                if gt_pack is None:
                    skipped_rows += 1
                    continue

                ground_truth, _ = gt_pack
                point_predictions = merge_channel_first_array(
                    coerce_2d_float_array(row.get("point_predictions")),
                    merge_plan,
                )
                metrics_output = _compute_binary_metrics_for_sample(
                    point_predictions=point_predictions,
                    ground_truth=ground_truth,
                    threshold=self.threshold,
                )
                record = {
                    "user_id": user_id,
                    "model": model_name,
                    "history_length": history_length,
                    "forecasting_length": forecast_length,
                    "f1": metrics_output["f1"],
                    "auroc": metrics_output["auroc"],
                    "auprc": metrics_output["auprc"],
                    "binary_valid_count": metrics_output["binary_valid_count"],
                    "binary_positive_count": metrics_output["binary_positive_count"],
                    "binary_negative_count": metrics_output["binary_negative_count"],
                }
                user_records.append(record)
                saved_rows += 1

            if user_records:
                records_by_user[user_id] = user_records
                computed_user_count += 1

        saved_files_by_metric = _save_binary_metrics_by_metric(
            output_root=output_run_dir,
            model_key=sanitize_name(model_name),
            records_by_user=records_by_user,
            metric_names=self.metric_names,
        )
        return {
            "run_key": self.run_key,
            "run_path": str(self.run_path),
            "model_name": model_name,
            "saved_rows": saved_rows,
            "skipped_rows": skipped_rows,
            "computed_user_count": computed_user_count,
            "threshold": self.threshold,
            "output_run_dir": str(output_run_dir),
            "saved_files_by_metric": saved_files_by_metric,
        }


class BinaryOfflineMetricsCalculator:
    """Orchestrate binary offline metric generation across forecasting runs."""

    def __init__(
        self,
        *,
        evaluation_result_paths: dict[str, str],
        metrics_output_path: str,
        threshold: float,
        max_user: int | None = None,
        combine_channels: bool = True,
        metric_names: tuple[str, ...] = _METRIC_NAMES,
    ):
        """Initialize binary metric generation across forecasting runs."""
        self.evaluation_result_paths = {
            str(name): Path(path) for name, path in evaluation_result_paths.items()
        }
        self.metrics_output_path = Path(metrics_output_path)
        self.metrics_output_path.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.max_user = int(max_user) if max_user is not None else None
        self.combine_channels = bool(combine_channels)
        self.metric_names = tuple(metric_names)

    def run(self) -> dict[str, Any]:
        """Run binary metric generation for all configured runs."""
        run_summaries: list[dict[str, Any]] = []
        for run_key, run_path in self.evaluation_result_paths.items():
            if not run_path.exists():
                logger.warning("Run path does not exist for key=%s, skipped: %s", run_key, run_path)
                continue

            logger.info("Processing binary metrics for key=%s run path=%s", run_key, run_path)
            pipeline = BinaryOfflineMetricsPipeline(
                run_key=run_key,
                run_path=run_path,
                metrics_output_path=self.metrics_output_path,
                threshold=self.threshold,
                max_user=self.max_user,
                combine_channels=self.combine_channels,
                metric_names=self.metric_names,
            )
            run_summaries.append(pipeline.run())

        return {
            "runs": run_summaries,
            "total_runs": len(run_summaries),
            "total_saved_rows": int(sum(summary["saved_rows"] for summary in run_summaries)),
            "total_skipped_rows": int(sum(summary["skipped_rows"] for summary in run_summaries)),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for binary offline metric calculation."""
    parser = argparse.ArgumentParser(
        description="Compute per-sample binary forecasting metrics from saved forecasting outputs.",
    )
    parser.add_argument(
        "--evaluation-result-paths",
        "--run-dirs",
        nargs="+",
        default=[],
        help=(
            "One or more named forecasting run mappings in key=path format. "
            "Each path should point to one model output directory containing "
            "config.yaml and user parquet files."
        ),
    )
    parser.add_argument(
        "--metrics-output-path",
        "--output-dir",
        default="/home/lp925/code/MHC-benchmark/results/metrics",
        help="Root output directory to store offline metrics results.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize continuous scores for F1.",
    )
    parser.add_argument(
        "--max-user",
        type=int,
        default=None,
        help="Optional sequential cap on how many user parquet files to compute per run.",
    )
    return parser


def main() -> None:
    """Run binary offline metric calculation from CLI arguments."""
    args = build_parser().parse_args()
    if not args.evaluation_result_paths:
        raise ValueError("Please provide --evaluation-result-paths in key=path format")
    evaluation_result_paths = _parse_named_paths(args.evaluation_result_paths)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    calculator = BinaryOfflineMetricsCalculator(
        evaluation_result_paths=evaluation_result_paths,
        metrics_output_path=args.metrics_output_path,
        threshold=float(args.f1_threshold),
        max_user=args.max_user,
    )
    summary = calculator.run()
    compact_summary = {
        "total_runs": summary.get("total_runs"),
        "total_saved_rows": summary.get("total_saved_rows"),
        "total_skipped_rows": summary.get("total_skipped_rows"),
        "runs": [
            {
                "run_key": run_summary.get("run_key"),
                "model_name": run_summary.get("model_name"),
                "saved_rows": run_summary.get("saved_rows"),
                "skipped_rows": run_summary.get("skipped_rows"),
                "computed_user_count": run_summary.get("computed_user_count"),
                "threshold": run_summary.get("threshold"),
                "output_run_dir": run_summary.get("output_run_dir"),
            }
            for run_summary in summary.get("runs", [])
        ],
    }
    logging.getLogger(__name__).info("Binary offline metrics finished: %s", compact_summary)


if __name__ == "__main__":
    main()
