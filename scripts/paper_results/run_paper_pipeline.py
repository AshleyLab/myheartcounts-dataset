#!/usr/bin/env python
r"""End-to-end driver for the downstream paper-metrics pipeline.

Chains the bootstrap phases into one command — each phase is a subprocess, so a
failing phase prints its command and aborts:

  Phase 1 — ``bootstrap_downstream_draws.py``         → ``bootstrap_draws.parquet``
  Phase 2 — ``aggregate_downstream_paper_metrics.py`` → 4 sidecar CSVs
            ``aggregate_fairness_skill_score.py``     → ``fairness_skill_score_bootstrap.csv``

Predictions (per-(method, task) ``test.parquet`` under ``--predictions_dir``) come
from the eval pipeline run with ``--output.save_predictions true``; this driver
does not run the eval itself.

Usage::

    PYTHONPATH=src python scripts/paper_results/run_paper_pipeline.py \
        --predictions_dir results/eval/final/predictions \
        --csvs_dir results/eval/final \
        --output-dir results/paper \
        --methods stat_simple multirocket mae_encoder toto_encoder \
                  chronos2_encoder fe_xgboost hybrid_ssl_stat_simple gru_d_multitask \
        --baseline stat_simple --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run a subprocess (inheriting env, e.g. PYTHONPATH); raise on non-zero exit."""
    log.info("$ %s", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Command failed ({res.returncode}): {' '.join(str(c) for c in cmd)}")


def main() -> None:
    """Chain phase 1 → phase 2 (+ fairness reducer)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions_dir", type=Path, required=True)
    p.add_argument("--csvs_dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--baseline", default="stat_simple")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase 1 (assume bootstrap_draws.parquet already exists).",
    )
    p.add_argument(
        "--skip-phase2", action="store_true", help="Skip phase 2 (only produce the draws parquet)."
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draws = args.output_dir / "bootstrap_draws.parquet"
    py = [sys.executable]

    if not args.skip_phase1:
        _run(
            py
            + [
                str(HERE / "bootstrap_downstream_draws.py"),
                "--predictions_dir",
                str(args.predictions_dir),
                "--csvs_dir",
                str(args.csvs_dir),
                "--methods",
                *args.methods,
                "--n_bootstrap",
                str(args.n_bootstrap),
                "--seed",
                str(args.seed),
                "--output",
                str(draws),
            ],
            args.dry_run,
        )

    if not args.skip_phase2:
        _run(
            py
            + [
                str(HERE / "aggregate_downstream_paper_metrics.py"),
                "--draws",
                str(draws),
                "--output-dir",
                str(args.output_dir),
                "--baseline-method",
                args.baseline,
            ],
            args.dry_run,
        )
        _run(
            py
            + [
                str(HERE / "aggregate_fairness_skill_score.py"),
                "--draws",
                str(draws),
                "--output",
                str(args.output_dir / "fairness_skill_score_bootstrap.csv"),
                "--baseline-method",
                args.baseline,
            ],
            args.dry_run,
        )

    log.info("Pipeline complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
