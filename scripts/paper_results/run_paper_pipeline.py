#!/usr/bin/env python
r"""End-to-end driver for the imputation paper-metrics pipeline.

Reproduces the cross-imputer paper numbers in three stages:

1. **Phase 0** — for each method in the sweep config, run ``mhc-impute-eval``
   pinned to a per-method output directory (so the resulting ``pairs/``
   subdirectory is easy to locate).
2. Build a JSON manifest mapping ``{method: pairs_dir}`` from the per-method
   output directories.
3. **Phase 1** — invoke ``bootstrap_imputation_draws.py`` to produce
   ``bootstrap_draws.parquet``.
4. **Phase 2** — invoke ``aggregate_imputation_paper_metrics.py`` to produce
   the four sidecar CSVs, followed by ``aggregate_fairness_skill_score.py``
   to produce ``fairness_skill_score_bootstrap.csv`` (per-attribute and
   macro-averaged fairness skill scores; see the script's docstring for the
   formulation).

The driver is intentionally minimal: each phase is a subprocess. If a method
or phase fails, the failing command is printed and the driver exits
non-zero so the user can resume manually.

Usage::

    python scripts/paper_results/run_paper_pipeline.py \
        --sweep-config configs/paper/sweep_methods.yaml

Stages can be skipped (e.g. to re-run aggregation only)::

    python scripts/paper_results/run_paper_pipeline.py \
        --sweep-config configs/paper/sweep_methods.yaml \
        --skip-eval --skip-phase1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve repo root from this script's location.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the imputation paper pipeline end-to-end.")
    p.add_argument(
        "--sweep-config", type=Path, required=True,
        help="Path to sweep_methods.yaml describing methods and phase knobs",
    )
    p.add_argument(
        "--skip-eval", action="store_true",
        help="Skip phase 0 (assume per-method pairs/ already exist under runs_root)",
    )
    p.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip phase 1 (assume bootstrap_draws.parquet already exists)",
    )
    p.add_argument(
        "--skip-phase2", action="store_true",
        help="Skip phase 2 (only sweep + draws)",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Restrict to a subset of methods listed in the sweep config",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands that would be executed; do not run them",
    )
    return p.parse_args()


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run a subprocess; raise on non-zero exit."""
    pretty = " ".join(cmd)
    logger.info("$ %s", pretty)
    if dry_run:
        return
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        logger.error("Command failed (exit=%d): %s", res.returncode, pretty)
        sys.exit(res.returncode)


def _phase0_run_methods(cfg: dict, methods: list[dict], dry_run: bool) -> dict[str, Path]:
    """Run mhc-impute-eval per method; return {method: pairs_dir}."""
    runs_root = Path(cfg["runs_root"])
    common_overrides = cfg.get("common_overrides", []) or []
    method_dirs: dict[str, Path] = {}
    for m in methods:
        name = m["name"]
        run_dir = runs_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mhc-impute-eval",
            f"method={name}",
            f"hydra.run.dir={run_dir}",
            *common_overrides,
            *(m.get("overrides", []) or []),
        ]
        _run(cmd, dry_run)
        method_dirs[name] = run_dir / "pairs"
    return method_dirs


def _write_manifest(method_dirs: dict[str, Path], manifest_path: Path, dry_run: bool) -> None:
    manifest = {m: str(p.resolve()) for m, p in method_dirs.items() if p.exists()}
    missing = sorted(set(method_dirs) - set(manifest))
    if missing:
        logger.warning("Manifest skipping methods with no pairs/ dir: %s", missing)
    if not manifest:
        logger.error("No method has a pairs/ dir at the expected location; aborting")
        sys.exit(2)
    logger.info("Writing manifest with %d methods to %s", len(manifest), manifest_path)
    if dry_run:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _phase1_bootstrap(cfg: dict, dry_run: bool) -> None:
    script = REPO_ROOT / "scripts" / "paper_results" / "bootstrap_imputation_draws.py"
    cmd = [
        sys.executable, str(script),
        "--method-dirs", cfg["manifest_path"],
        "--output", cfg["draws_path"],
        "--n-boot", str(cfg["n_boot"]),
        "--seed", str(cfg["seed"]),
        "--splits", *cfg["splits"],
        "--age-bins", *[str(b) for b in cfg.get("age_bins", [18, 30, 40, 50, 60])],
    ]
    if not cfg.get("include_fairness", True):
        cmd.append("--no-fairness")
    if not cfg.get("include_auc", True):
        cmd.append("--no-auc")
    if cfg.get("exclude_unknown", False):
        cmd.append("--exclude-unknown")
    _run(cmd, dry_run)


def _phase2_aggregate(cfg: dict, dry_run: bool) -> None:
    script = REPO_ROOT / "scripts" / "paper_results" / "aggregate_imputation_paper_metrics.py"
    cmd = [
        sys.executable, str(script),
        "--draws", cfg["draws_path"],
        "--output-dir", cfg["output_root"],
        "--baseline-method", cfg["baseline_method"],
        "--clip-lower", str(cfg["clip_lower"]),
        "--clip-upper", str(cfg["clip_upper"]),
        "--lambda-fairness", str(cfg["lambda_fairness"]),
        "--fairness-combine", cfg["fairness_combine"],
        "--ci-level", str(cfg["ci_level"]),
    ]
    for d in cfg.get("disparity_fns", []) or []:
        cmd.extend(["--disparity-fn", d])
    _run(cmd, dry_run)


def _phase2_fairness_skill_score(cfg: dict, dry_run: bool) -> None:
    """Sidecar reducer: fairness skill score (per-attribute and macro-averaged).

    Independent of the four CSVs produced by ``_phase2_aggregate`` — reads
    the same Phase 1 draws and writes a single
    ``fairness_skill_score_bootstrap.csv`` under ``output_root``. Reuses the
    same clip bounds, baseline method, and CI level as the regular skill
    score for cross-table consistency.
    """
    script = REPO_ROOT / "scripts" / "paper_results" / "aggregate_fairness_skill_score.py"
    output_path = Path(cfg["output_root"]) / "fairness_skill_score_bootstrap.csv"
    cmd = [
        sys.executable, str(script),
        "--draws", cfg["draws_path"],
        "--output", str(output_path),
        "--baseline-method", cfg["baseline_method"],
        "--clip-lower", str(cfg["clip_lower"]),
        "--clip-upper", str(cfg["clip_upper"]),
        "--ci-level", str(cfg["ci_level"]),
    ]
    _run(cmd, dry_run)


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    cfg = yaml.safe_load(args.sweep_config.read_text())

    methods = cfg["methods"]
    if args.methods:
        methods = [m for m in methods if m["name"] in args.methods]
        if not methods:
            logger.error("--methods left no entries")
            return 2

    # Phase 0
    if not args.skip_eval:
        method_dirs = _phase0_run_methods(cfg, methods, args.dry_run)
    else:
        runs_root = Path(cfg["runs_root"])
        method_dirs = {m["name"]: runs_root / m["name"] / "pairs" for m in methods}

    # Build manifest
    _write_manifest(method_dirs, Path(cfg["manifest_path"]), args.dry_run)

    # Phase 1
    if not args.skip_phase1:
        _phase1_bootstrap(cfg, args.dry_run)

    # Phase 2
    if not args.skip_phase2:
        _phase2_aggregate(cfg, args.dry_run)
        _phase2_fairness_skill_score(cfg, args.dry_run)

    logger.info("Done. Sidecar CSVs in %s", cfg["output_root"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
