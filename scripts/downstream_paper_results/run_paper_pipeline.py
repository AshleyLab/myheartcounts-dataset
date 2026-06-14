#!/usr/bin/env python
r"""Config-driven driver for the downstream paper-metrics pipeline.

One command, one config: reads ``configs/paper/downstream_paper.yaml`` (or any
``--config``) and chains the bootstrap phases, each as a subprocess so a failing
phase prints its command and aborts:

  Phase 1 — ``bootstrap_downstream_draws.py``         → ``bootstrap_draws.parquet``
  Phase 2 — ``aggregate_downstream_paper_metrics.py`` → 4 sidecar CSVs
            ``aggregate_fairness_skill_score.py``     → ``fairness_skill_score_bootstrap.csv``

Predictions (per-(method, task) ``test.parquet`` under ``predictions_dir``) come
from the eval run with ``PREDICTIONS_DIR`` set; this driver does not run the eval.

Usage::

    PYTHONPATH=src python scripts/downstream_paper_results/run_paper_pipeline.py \
        --config configs/paper/downstream_paper.yaml

    # re-aggregate only (draws already exist), e.g. to retune the fairness knobs:
    ... --config configs/paper/downstream_paper.yaml --skip-phase1
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run a subprocess (inheriting env, e.g. PYTHONPATH); raise on non-zero exit."""
    log.info("$ %s", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    if subprocess.run(cmd, check=False).returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(str(c) for c in cmd)}")


def _phase1_bootstrap(cfg: dict, draws: Path, methods: list[str], dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(HERE / "bootstrap_downstream_draws.py"),
            "--predictions_dir", str(cfg["predictions_dir"]),
            "--csvs_dir", str(cfg["csvs_dir"]),
            "--methods", *methods,
            "--n_bootstrap", str(cfg["n_bootstrap"]),
            "--seed", str(cfg["seed"]),
            "--fairness_attributes", *cfg.get("fairness_attributes", ["age_group", "sex"]),
            "--output", str(draws),
        ],
        dry_run,
    )


def _phase2_aggregate(cfg: dict, draws: Path, out_dir: Path, dry_run: bool) -> None:
    agg = [
        sys.executable,
        str(HERE / "aggregate_downstream_paper_metrics.py"),
        "--draws", str(draws),
        "--output-dir", str(out_dir),
        "--baseline-method", cfg["baseline_method"],
        "--clip-lower", str(cfg["clip_lower"]),
        "--clip-upper", str(cfg["clip_upper"]),
        "--lambda-fairness", str(cfg["lambda_fairness"]),
        "--fairness-combine", cfg["fairness_combine"],
        "--ci-level", str(cfg["ci_level"]),
    ]
    for d in cfg.get("disparity_fns") or []:
        agg += ["--disparity-fn", d]
    _run(agg, dry_run)
    _run(
        [
            sys.executable,
            str(HERE / "aggregate_fairness_skill_score.py"),
            "--draws", str(draws),
            "--output", str(out_dir / "fairness_skill_score_bootstrap.csv"),
            "--baseline-method", cfg["baseline_method"],
            "--clip-lower", str(cfg["clip_lower"]),
            "--clip-upper", str(cfg["clip_upper"]),
            "--ci-level", str(cfg["ci_level"]),
        ],
        dry_run,
    )


def main() -> None:
    """Read the config, chain phase 1 → phase 2 (+ fairness reducer)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pipeline config YAML (see configs/paper/downstream_paper.yaml).",
    )
    p.add_argument(
        "--methods", nargs="+", default=None, help="Restrict to a subset of the config's methods."
    )
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase 1 (reuse the existing bootstrap_draws.parquet).",
    )
    p.add_argument(
        "--skip-phase2", action="store_true", help="Skip phase 2 (only build the draws parquet)."
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any method has no predictions dir — for runs whose numbers are published.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the phase commands without running.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    methods = args.methods or cfg["methods"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = out_dir / "bootstrap_draws.parquet"

    if args.strict:
        pred_root = Path(cfg["predictions_dir"])
        missing = [m for m in methods if not (pred_root / m).is_dir()]
        if missing:
            raise SystemExit(f"[strict] methods with no predictions dir under {pred_root}: {missing}")

    if not args.skip_phase1:
        _phase1_bootstrap(cfg, draws, methods, args.dry_run)
    if not args.skip_phase2:
        _phase2_aggregate(cfg, draws, out_dir, args.dry_run)

    log.info("Pipeline complete → %s", out_dir)


if __name__ == "__main__":
    main()
