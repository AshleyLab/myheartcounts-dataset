"""Build + ship the frozen Seasonal-Naive per-user substrate for the public API.

``openmhc.evaluate_forecasting`` computes a user's skill score against this
baseline, so it must be faithful to the paper's ``seasonal_naive`` run. Built from
the canonical seasonal_naive metric trees (the same trees the leaderboard run
aggregates) with the paper-default scored set (mae + auroc), micro/user. Writes:

    src/openmhc/data/baselines/forecasting_seasonal_naive_per_user_errors.parquet

Usage::

    python scripts/paper_results/forecasting/parity/produce_seasonal_naive_substrate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402
from forecasting_evaluation.metrics.per_user_errors import (  # noqa: E402
    build_per_user_metrics,
    write_per_user_metrics_parquet,
)

DEFAULT_SUMMARY = REPO_ROOT / "results/forecasting_eval/simurgh/summary/forecasting_bca_20260618"
DEFAULT_OUT = (
    SRC_ROOT / "openmhc/data/baselines/forecasting_seasonal_naive_per_user_errors.parquet"
)


def main() -> int:
    """Build the seasonal_naive substrate and write it to the shipped baseline path."""
    p = argparse.ArgumentParser(description="Ship the seasonal_naive per-user baseline substrate.")
    p.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="seasonal_naive metrics dir (overrides the summary-dir lookup)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if args.metrics_root is not None:
        metrics_root = str(args.metrics_root)
    else:
        models = json.loads((args.summary_dir / "skill_rank_models.json").read_text())
        models = models.get("models", models)
        if _spec.PAPER_BASELINE not in models:
            raise SystemExit(
                f"{_spec.PAPER_BASELINE!r} not in {args.summary_dir}/skill_rank_models.json"
            )
        metrics_root = models[_spec.PAPER_BASELINE]
        if not Path(metrics_root).is_absolute():  # repo-relative paths in the json
            metrics_root = str(REPO_ROOT / metrics_root)

    df = build_per_user_metrics(
        models={_spec.PAPER_BASELINE: {"path": metrics_root, "display_name": _spec.PAPER_BASELINE}},
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
    )
    if df.empty:
        raise SystemExit(f"empty substrate from {metrics_root} — check the metric trees")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_per_user_metrics_parquet(
        df,
        args.out,
        meta={
            "method": _spec.PAPER_BASELINE,
            "source_metrics_root": metrics_root,
            "within_user_aggregation": "micro",
            "aggregation_unit": "user",
            "continuous_metrics": list(_spec.PAPER_CONTINUOUS_METRICS),
            "binary_metrics": list(_spec.PAPER_BINARY_METRICS),
        },
    )
    print(f"Wrote {len(df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
