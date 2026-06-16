"""CLI entry point for offline forecasting metrics calculation."""

from __future__ import annotations

import argparse
import logging

from forecasting_evaluation.metrics.offline.runner import OfflineMetricsCalculator


def _parse_named_paths(items: list[str]) -> dict[str, str]:
    """Parse CLI items in key=path format."""
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

def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute offline forecasting metrics from saved forecasting run outputs.",
    )
    parser.add_argument(
        "--evaluation-result-paths",
        "--run-dirs",
        nargs="+",
        default=[],
        help=(
            "One or more named forecasting run mappings in key=path format. "
            "Each path should point to one model output directory containing "
            "config.yaml and user parquet files (e.g. results/forecasting_eval/<experiment>/<model>)."
        ),
    )
    parser.add_argument(
        "--metrics-output-path",
        "--output-dir",
        default="/home/lp925/code/MHC-benchmark/results/metrics",
        help="Root output directory to store offline metrics results.",
    )
    parser.add_argument(
        "--max-user",
        type=int,
        default=None,
        help=(
            "Optional sequential cap on how many user parquet metrics files to compute "
            "per run. Use None to process all users."
        ),
    )
    parser.add_argument(
        "--combine-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Merge paired phone/watch step and distance channels (nan-mean) before "
            "computing offline metrics. Default is per-task (channels kept separate, "
            "scored like sleep/workout); pass --combine-channels for the legacy "
            "signal-merged behaviour (appendix raw hour-group tables)."
        ),
    )
    return parser


def main() -> None:
    """Run offline forecasting metrics calculation from CLI arguments."""
    args = build_parser().parse_args()
    if not args.evaluation_result_paths:
        raise ValueError("Please provide --evaluation-result-paths in key=path format")
    evaluation_result_paths = _parse_named_paths(args.evaluation_result_paths)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    calculator = OfflineMetricsCalculator(
        evaluation_result_paths=evaluation_result_paths,
        metrics_output_path=args.metrics_output_path,
        max_user=args.max_user,
        combine_channels=args.combine_channels,
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
                "combine_channels": run_summary.get("combine_channels"),
                "output_run_dir": run_summary.get("output_run_dir"),
            }
            for run_summary in summary.get("runs", [])
        ],
    }
    logging.getLogger(__name__).info("Offline metrics finished: %s", compact_summary)


if __name__ == "__main__":
    main()
