#!/usr/bin/env python
r"""Phase 1 of the imputation paper-metrics bootstrap.

Reads each method's saved imputation pairs/, runs a paired participant-level
bootstrap **across methods** (shared resample matrix per (scenario, split,
subgroup) cell so inter-method differences are paired), and writes a
long-format Parquet of per-draw error values plus a sidecar metadata JSON.

Phase 2 (``aggregate_imputation_paper_metrics.py``) reads that Parquet and
produces summary CSVs (mean / SE / 95 % CI for skill score, average rank,
per-subgroup S_g, and any registered disparity / fairness-combine).

Example::

    python scripts/paper_results/bootstrap_imputation_draws.py \
        --method-dirs configs/paper/bootstrap_method_dirs.json \
        --output results/paper/bootstrap_draws.parquet \
        --n-boot 1000 --seed 42 \
        --splits test \
        --include-fairness

The JSON manifest maps method -> pairs_dir, e.g.::

    {
      "locf":            "results/imputation_eval/imputation_locf_X/pairs",
      "mae_daily_nodropout": "results/imputation_eval/mae_daily_nodropout_Y/pairs"
    }

Each pairs_dir must contain ``manifest_<split>.parquet`` and per-scenario
subdirs with ``<scenario>/<split>/pairs_ch{NN}.parquet``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    compute_per_draw_errors,
    write_draws_parquet,
)
from imputation_evaluation.evaluation.pair_writer import load_sample_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: build bootstrap_draws.parquet for the paper metrics pipeline",
    )
    p.add_argument(
        "--method-dirs", type=Path, required=True,
        help="JSON manifest mapping {method: pairs_dir}",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet (.parquet). Sidecar .meta.json written alongside.",
    )
    p.add_argument(
        "--n-boot", type=int, default=1000,
        help="Number of bootstrap draws (default 1000)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed; per-cell seeds derived deterministically",
    )
    p.add_argument(
        "--splits", nargs="+", default=["test"],
        help="Splits to process (default: test). Example: --splits test val",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Scenarios to process (default: auto-discover from first method's dir)",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Restrict to a subset of methods from --method-dirs (default: all)",
    )
    p.add_argument(
        "--no-auc", action="store_true",
        help="Skip AUC bootstrap for binary channels (faster, but no fairness for binary)",
    )
    p.add_argument(
        "--no-fairness", action="store_true",
        help="Skip subgroup demographic mapping (fairness CIs cannot be computed in phase 2)",
    )
    p.add_argument(
        "--age-bins", type=int, nargs="+", default=[18, 30, 40, 50, 60],
        help="Age-bin edges for the age_group attribute (default: 18 30 40 50 60)",
    )
    p.add_argument(
        "--exclude-unknown", action="store_true",
        help="Skip subgroup_value=='unknown' cells",
    )
    p.add_argument(
        "--channel-stds-path", type=Path, default=None,
        help="Override channel_stds.npy path (default: <first method dir>/channel_stds.npy)",
    )
    return p.parse_args()


def _discover_scenarios(method_dirs: dict[str, Path], split: str) -> list[str]:
    """Auto-discover scenarios = subdirs that contain a /<split>/ child."""
    seen: set[str] = set()
    for root in method_dirs.values():
        root = Path(root)
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / split).is_dir():
                seen.add(child.name)
    return sorted(seen)


def _build_subgroup_mapping(
    pairs_dir: Path, split: str, age_bins: list[int],
) -> dict[int, dict[str, str]] | None:
    """Build {sample_idx: {age_group, sex}} mapping from a method's manifest.

    Mirrors the inline pattern in ``imputation_evaluation.sensitivity``: openmhc's
    ``get_user_demographics`` returns ``{"birth_year": int | None, "sex": str}``
    per user (post the labels-api privacy migration), and
    ``years_between_birth_year(birth_year, sample_date)`` yields a whole-year
    age that we then pass through ``bin_age``.
    """
    manifest = load_sample_manifest(pairs_dir, split)
    if manifest is None:
        return None
    from imputation_evaluation.sensitivity import bin_age, get_user_demographics
    from labels.api import STORE, years_between_birth_year

    sample_idxs = manifest.column("sample_idx").to_numpy()
    user_ids = manifest.column("user_id").to_pylist()
    dates = manifest.column("date").to_pylist()
    unique_users = sorted(set(user_ids))
    logger.info(
        "[split=%s] looking up demographics for %d users …", split, len(unique_users),
    )
    user_demographics = get_user_demographics(STORE, unique_users)

    out: dict[int, dict[str, str]] = {}
    for sidx, uid, date_str in zip(sample_idxs, user_ids, dates):
        # ``get_user_demographics`` returns ``{"birth_year": int | None, "sex": str}``
        # after the labels-api privacy migration (commit 1e8795b on main). The
        # Dataverse ``enrollment_info.json`` ships year-precision birthdates,
        # which ``years_between_birth_year`` synthesizes into ages.
        demo = user_demographics.get(uid, {"birth_year": None, "sex": "unknown"})
        age_group = "unknown"
        birth_year = demo["birth_year"]
        if birth_year is not None:
            try:
                sample_date = pd.Timestamp(date_str)
                age = years_between_birth_year(birth_year, sample_date)
                age_group = bin_age(age, age_bins)
            except Exception:
                pass
        out[int(sidx)] = {"age_group": age_group, "sex": demo["sex"]}
    return out


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()

    with args.method_dirs.open() as f:
        raw_dirs = json.load(f)
    method_dirs: dict[str, Path] = {m: Path(p) for m, p in raw_dirs.items()}
    if args.methods:
        method_dirs = {m: p for m, p in method_dirs.items() if m in args.methods}
    if not method_dirs:
        logger.error("No methods left after --methods filter")
        return 2

    # Validate paths
    for m, p in list(method_dirs.items()):
        if not p.exists():
            logger.warning("method=%s: %s does not exist — skipping", m, p)
            method_dirs.pop(m)
    if not method_dirs:
        logger.error("All method dirs missing; aborting")
        return 2

    # Resolve scenarios
    if args.scenarios:
        scenarios = list(args.scenarios)
    else:
        # Use first split for discovery; downstream will skip cells absent for a split.
        scenarios = _discover_scenarios(method_dirs, args.splits[0])
    if not scenarios:
        logger.error("No scenarios discovered; aborting")
        return 2
    logger.info("Methods: %s", list(method_dirs.keys()))
    logger.info("Scenarios: %s", scenarios)
    logger.info("Splits: %s", args.splits)

    # Subgroup mapping: per-split, built from the first method's manifest
    # (manifests are scenario-independent and shared across methods).
    subgroup_mappings: dict[str, dict[int, dict[str, str]]] = {}
    if not args.no_fairness:
        ref_method = next(iter(method_dirs))
        ref_dir = method_dirs[ref_method]
        for split in args.splits:
            sg = _build_subgroup_mapping(ref_dir, split, args.age_bins)
            if sg is None:
                logger.warning(
                    "No manifest_%s.parquet at %s; skipping fairness for this split",
                    split, ref_dir,
                )
            else:
                subgroup_mappings[split] = sg
                logger.info("[split=%s] subgroup mapping built: %d samples",
                            split, len(sg))

    # Channel stds
    channel_stds = None
    if args.channel_stds_path:
        channel_stds = np.load(args.channel_stds_path)
        logger.info("Using channel_stds from %s", args.channel_stds_path)

    # Build per-draw errors
    t0 = datetime.now()
    df = compute_per_draw_errors(
        method_dirs={m: Path(p) for m, p in method_dirs.items()},
        scenarios=scenarios,
        splits=args.splits,
        n_boot=args.n_boot,
        seed=args.seed,
        subgroup_mappings=subgroup_mappings if subgroup_mappings else None,
        channel_stds=channel_stds,
        include_auc=not args.no_auc,
        exclude_unknown=args.exclude_unknown,
    )
    t_elapsed = (datetime.now() - t0).total_seconds()
    logger.info("Per-draw errors: %d rows in %.1fs", len(df), t_elapsed)

    if df.empty:
        logger.error("No errors produced; aborting before write")
        return 1

    meta = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "splits": args.splits,
        "scenarios": scenarios,
        "methods": list(method_dirs.keys()),
        "method_dirs": {m: str(p) for m, p in method_dirs.items()},
        "include_auc": not args.no_auc,
        "include_fairness": not args.no_fairness,
        "age_bins": args.age_bins,
        "exclude_unknown": args.exclude_unknown,
        "elapsed_seconds": t_elapsed,
        "n_rows": len(df),
        "git_commit": _git_commit(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    write_draws_parquet(df, args.output, meta=meta)
    logger.info("Wrote %s and %s.meta.json", args.output, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
