#!/usr/bin/env python
r"""Deterministic point estimates for the imputation paper's headline metrics.

Reads each method's saved ``pairs/`` directories (same input as
``bootstrap_imputation_draws.py``) and emits the three deterministic
sidecar CSVs that the bootstrap pipeline summarises with mean / SE / CI:

* ``skill_scores.csv``         — :func:`paper_metrics_core.compute_skill_scores`
* ``avg_rankings.csv``         — :func:`paper_metrics_core.compute_average_rankings`
* ``fairness_skill_scores.csv` — :func:`paper_metrics_core.compute_fair_skill_scores`

These are the *point estimates* the leaderboard and paper quote.
``aggregate_imputation_paper_metrics.py`` (Phase 2 of the bootstrap)
shares the same kernels and produces the corresponding ``*_bootstrap.csv``
files; the bootstrap mean should match the point estimate up to resample
noise (the parity is enforced in
``tests/imputation_evaluation/test_paper_metrics_core.py``).

Example::

    python scripts/paper_results/compute_imputation_paper_metrics.py \
        --method-dirs configs/paper/bootstrap_method_dirs.json \
        --output-dir results/paper/ \
        --splits test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS

from imputation_evaluation.evaluation.pair_aggregator import (
    aggregate_pairs,
    aggregate_pairs_by_subgroup,
)
from imputation_evaluation.evaluation.pair_writer import load_sample_manifest
from imputation_evaluation.evaluation.paper_metrics_core import (
    BASELINE_CONTINUOUS,
    CLIP_LOWER,
    CLIP_UPPER,
    DEFAULT_FAIRNESS_ATTRS,
    build_baseline_errors,
    compute_average_rankings,
    compute_fair_skill_scores,
    compute_skill_scores,
    extract_errors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute deterministic point estimates (skill score, average rank, "
            "fair skill score) from saved pairs/ directories."
        ),
    )
    p.add_argument(
        "--method-dirs", type=Path, required=True,
        help="JSON manifest mapping {method: pairs_dir} (same shape as bootstrap_imputation_draws.py)",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for skill_scores.csv / avg_rankings.csv / fairness_skill_scores.csv",
    )
    p.add_argument(
        "--splits", nargs="+", default=["test"],
        help="Splits to process (default: test)",
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
        "--baseline-method", default=BASELINE_CONTINUOUS,
        help=f"Method to treat as the baseline (default: {BASELINE_CONTINUOUS})",
    )
    p.add_argument(
        "--clip-lower", type=float, default=CLIP_LOWER,
        help=f"Lower clip bound for error ratios (default: {CLIP_LOWER})",
    )
    p.add_argument(
        "--clip-upper", type=float, default=CLIP_UPPER,
        help=f"Upper clip bound for error ratios (default: {CLIP_UPPER})",
    )
    p.add_argument(
        "--attrs", nargs="+", default=list(DEFAULT_FAIRNESS_ATTRS),
        help=f"Sensitive attributes for the fair skill score (default: {' '.join(DEFAULT_FAIRNESS_ATTRS)})",
    )
    p.add_argument(
        "--age-bins", type=int, nargs="+", default=[18, 30, 40, 50, 60],
        help="Age-bin edges for the age_group attribute (default: 18 30 40 50 60)",
    )
    p.add_argument(
        "--exclude-unknown", action="store_true",
        help="Skip subgroup_value=='unknown' cells (default: include, matches bootstrap)",
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

    Mirrors ``bootstrap_imputation_draws.py::_build_subgroup_mapping`` so the
    deterministic and bootstrap pipelines see the same demographic bucketing.
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


def _channel_type(ch_idx: int) -> str:
    return "continuous" if ch_idx in CONTINUOUS_CHANNEL_INDICES else "binary"


def _per_channel_to_rows(
    per_channel: dict[str, dict],
    *,
    method: str,
    scenario: str,
    split: str,
    subgroup_attr: str,
    subgroup_value: str,
) -> list[dict]:
    """Flatten a ``per_channel`` metrics dict to long-format registry rows."""
    rows: list[dict] = []
    for ch in range(N_CHANNELS):
        ch_key = f"ch_{ch}"
        m = per_channel.get(ch_key, {})
        rows.append({
            "method": method,
            "scenario": scenario,
            "split": split,
            "channel": ch_key,
            "channel_type": _channel_type(ch),
            "subgroup_attr": subgroup_attr,
            "subgroup_value": subgroup_value,
            # extract_errors() reads nRMSE for continuous channels and roc_auc
            # for binary ones (E = 1 - AUC), so we surface both with NaN
            # fallback for channels that had no masked positions.
            "nRMSE": float(m.get("normalized_rmse", np.nan)),
            "roc_auc": float(m.get("roc_auc", np.nan)),
        })
    return rows


def _gather_registry(
    method_dirs: dict[str, Path],
    *,
    scenarios: list[str],
    splits: list[str],
    age_bins: list[int],
    exclude_unknown: bool,
    channel_stds_path: Path | None,
) -> pd.DataFrame:
    """Build the long-format errors registry for every (method, scenario, split, subgroup) cell."""
    methods = list(method_dirs.keys())

    # ------------------ resolve channel_stds ------------------
    if channel_stds_path is None:
        channel_stds_path = Path(method_dirs[methods[0]]) / "channel_stds.npy"
    channel_stds = np.load(channel_stds_path).astype(np.float64)
    if channel_stds.shape[0] < N_CHANNELS:
        raise ValueError(
            f"channel_stds has {channel_stds.shape[0]} entries, need {N_CHANNELS}"
        )

    rows: list[dict] = []

    # Subgroup mapping is per-split, scenario-independent; build once per split
    # from the first method's manifest (mirrors bootstrap_imputation_draws.py).
    subgroup_mappings: dict[str, dict[int, dict[str, str]]] = {}
    for split in splits:
        ref_root = Path(method_dirs[methods[0]])
        mapping = _build_subgroup_mapping(ref_root, split, age_bins)
        if mapping is None:
            logger.warning(
                "[split=%s] could not load manifest from %s — fairness rows will be empty",
                split, ref_root,
            )
            subgroup_mappings[split] = {}
        else:
            subgroup_mappings[split] = mapping

    for method, root in method_dirs.items():
        root = Path(root)
        for split in splits:
            mapping = subgroup_mappings.get(split, {})
            for scenario in scenarios:
                ssd = root / scenario / split
                if not ssd.exists():
                    logger.info("method=%s scenario=%s split=%s: %s missing — skipping",
                                method, scenario, split, ssd)
                    continue

                # "all / all" cell — deterministic per-channel metrics.
                metrics_all = aggregate_pairs(ssd, channel_stds)
                rows.extend(_per_channel_to_rows(
                    metrics_all.get("per_channel", {}),
                    method=method, scenario=scenario, split=split,
                    subgroup_attr="all", subgroup_value="all",
                ))

                # Per-subgroup cells — keyed by (attr, subgroup_value).
                if mapping:
                    per_sg = aggregate_pairs_by_subgroup(ssd, channel_stds, mapping)
                    for attr, groups in per_sg.items():
                        for group_name, metrics_g in groups.items():
                            if exclude_unknown and group_name == "unknown":
                                continue
                            rows.extend(_per_channel_to_rows(
                                metrics_g.get("per_channel", {}),
                                method=method, scenario=scenario, split=split,
                                subgroup_attr=attr, subgroup_value=group_name,
                            ))

    return pd.DataFrame(rows)


def _build_errors_long(
    registry: pd.DataFrame,
    *,
    splits: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (errors_all, errors_by_subgroup) from the registry.

    ``errors_all`` covers the global ``("all", "all")`` cell and feeds the
    skill-score / average-rank kernels. ``errors_by_subgroup`` carries the
    demographic cells and feeds the fair-skill kernel.
    """
    errors_all_frames: list[pd.DataFrame] = []
    errors_sg_frames: list[pd.DataFrame] = []
    for split in splits:
        df = registry[registry["split"] == split]
        if df.empty:
            continue
        all_cell = extract_errors(df, split=split, subgroup_attr="all", subgroup_value="all")
        if not all_cell.empty:
            all_cell["split"] = split
            errors_all_frames.append(all_cell)

        # Build the long-format subgroup frame manually (extract_errors only
        # filters one (attr, value) pair at a time; we want every demographic
        # cell stacked together with its subgroup columns preserved).
        sg = df[df["subgroup_attr"] != "all"].copy()
        if not sg.empty:
            sg_long = []
            for (attr, value), grp in sg.groupby(["subgroup_attr", "subgroup_value"], observed=True):
                sub = extract_errors(grp, split=split, subgroup_attr=attr, subgroup_value=value)
                if sub.empty:
                    continue
                sub["subgroup_attr"] = attr
                sub["subgroup_value"] = value
                sub["split"] = split
                sg_long.append(sub)
            if sg_long:
                errors_sg_frames.append(pd.concat(sg_long, ignore_index=True))

    errors_all = (
        pd.concat(errors_all_frames, ignore_index=True)
        if errors_all_frames else pd.DataFrame(
            columns=["method", "scenario", "channel", "channel_type", "E", "split"]
        )
    )
    errors_sg = (
        pd.concat(errors_sg_frames, ignore_index=True)
        if errors_sg_frames else pd.DataFrame(
            columns=[
                "method", "scenario", "channel", "channel_type", "E",
                "subgroup_attr", "subgroup_value", "split",
            ]
        )
    )
    return errors_all, errors_sg


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.method_dirs.open() as f:
        raw_dirs = json.load(f)
    method_dirs: dict[str, Path] = {m: Path(p) for m, p in raw_dirs.items()}
    if args.methods:
        method_dirs = {m: p for m, p in method_dirs.items() if m in args.methods}
    if not method_dirs:
        logger.error("No methods left after --methods filter")
        return 2

    for m, p in list(method_dirs.items()):
        if not p.exists():
            logger.warning("method=%s: %s does not exist — skipping", m, p)
            method_dirs.pop(m)
    if not method_dirs:
        logger.error("All method dirs missing; aborting")
        return 2

    if args.scenarios:
        scenarios = list(args.scenarios)
    else:
        scenarios = _discover_scenarios(method_dirs, args.splits[0])
    if not scenarios:
        logger.error("No scenarios discovered; aborting")
        return 2
    logger.info("Methods: %s", list(method_dirs.keys()))
    logger.info("Scenarios: %s", scenarios)
    logger.info("Splits: %s", args.splits)

    registry = _gather_registry(
        method_dirs,
        scenarios=scenarios,
        splits=args.splits,
        age_bins=args.age_bins,
        exclude_unknown=args.exclude_unknown,
        channel_stds_path=args.channel_stds_path,
    )
    logger.info("Registry rows: %d", len(registry))
    if registry.empty:
        logger.error("Registry is empty; aborting")
        return 2

    errors_all, errors_sg = _build_errors_long(registry, splits=args.splits)

    # ------------------------------------------------------------------
    # Skill score + average rank — per split, on the "all/all" cell.
    # ------------------------------------------------------------------
    skill_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    for split in args.splits:
        ea = errors_all[errors_all["split"] == split].drop(columns=["split"], errors="ignore")
        if ea.empty:
            continue
        baseline = build_baseline_errors(
            ea,
            baseline_continuous=args.baseline_method,
            baseline_binary=args.baseline_method,
        )
        skill = compute_skill_scores(
            ea, baseline,
            clip_lower=args.clip_lower, clip_upper=args.clip_upper,
        )
        skill["split"] = split
        skill_frames.append(skill)

        rank = compute_average_rankings(ea)
        rank["split"] = split
        rank_frames.append(rank)

    # ------------------------------------------------------------------
    # Fair skill score — per split, on the per-subgroup cells.
    # ------------------------------------------------------------------
    fair_frames: list[pd.DataFrame] = []
    for split in args.splits:
        es = errors_sg[errors_sg["split"] == split].drop(columns=["split"], errors="ignore")
        if es.empty:
            continue
        fair = compute_fair_skill_scores(
            es,
            attrs=args.attrs,
            baseline_method=args.baseline_method,
            clip_lower=args.clip_lower,
            clip_upper=args.clip_upper,
        )
        fair["split"] = split
        fair_frames.append(fair)

    out_paths = {
        "skill_scores":         args.output_dir / "skill_scores.csv",
        "avg_rankings":         args.output_dir / "avg_rankings.csv",
        "fairness_skill_scores": args.output_dir / "fairness_skill_scores.csv",
    }
    frames = {
        "skill_scores": skill_frames,
        "avg_rankings": rank_frames,
        "fairness_skill_scores": fair_frames,
    }
    for key, path in out_paths.items():
        tbl = (
            pd.concat(frames[key], ignore_index=True)
            if frames[key] else pd.DataFrame()
        )
        tbl.to_csv(path, index=False, float_format="%.6f")
        logger.info("Wrote %s (%d rows)", path, len(tbl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
