"""§1 LOCF per-user-errors parity test.

The first gate of the imputation-metrics refactor: the new canonical
producer :func:`build_per_user_errors` must reproduce today's
``per_user_errors.parquet`` (as written by
:func:`compute_per_draw_errors` with ``emit_per_user_errors=True``)
exactly for the LOCF slice. Every downstream skill / fairness / rank
number is paired against LOCF; if this gate fails the whole refactor
must stop.

This script:

  1. Re-builds LOCF's per-user-errors via the new producer, against the
     existing on-disk pairs at ``/scratch/.../runs/locf/pairs``.
  2. Writes the result to a sibling verification dir
     (``/scratch/.../paper-verification/locf_per_user_errors.parquet``)
     — never to ``paper/``.
  3. Outer-joins against the ground-truth slice
     (``paper/per_user_errors.parquet[method=='locf']``) on the 8 keys
     and asserts:
       * Row-set equality (no rows only in new, none only in ground truth).
       * ``E_per_user`` byte-equality after a ``float32`` cast (parquet
         stores the column as float32 already, so the cast is idempotent
         for already-loaded data).
  4. Exits non-zero on any mismatch, printing the first few offending
     rows for debugging.

Run via the sibling ``parity_locf.sbatch``; per Sherlock policy this
must be a SLURM job (no Python on the login node).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.pair_writer import load_sample_manifest
from imputation_evaluation.evaluation.per_user_errors import (
    PER_USER_ERRORS_PARQUET_COLUMNS,
    build_per_user_errors,
    write_per_user_errors_parquet,
)

logger = logging.getLogger(__name__)


# Default inputs match the existing on-disk run (see paper/per_user_errors.parquet.meta.json).
DEFAULT_LOCF_PAIRS = Path("/scratch/users/schuetzn/openmhc-imputation-eval/runs/locf/pairs")
DEFAULT_GT_PARQUET = Path(
    "/scratch/users/schuetzn/openmhc-imputation-eval/paper/per_user_errors.parquet"
)
DEFAULT_OUT_PARQUET = Path(
    "/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification/locf_per_user_errors.parquet"
)
DEFAULT_SPLITS = ["test"]
DEFAULT_SCENARIOS = [
    "intensity_failure",
    "random_noise",
    "signal_slice",
    "sleep_gap",
    "temporal_slice",
    "workout_gap",
]
DEFAULT_AGE_BINS = [18, 30, 40, 50, 60]


JOIN_KEYS = [
    "method",
    "scenario",
    "split",
    "channel",
    "channel_type",
    "subgroup_attr",
    "subgroup_value",
    "user_id",
]


def _build_subgroup_mapping(
    pairs_dir: Path,
    split: str,
    age_bins: list[int],
) -> dict[int, dict[str, str]] | None:
    """Mirror ``bootstrap_imputation_draws.py::_build_subgroup_mapping``.

    Vendored here so the parity test is self-contained — the original
    lives in a scripts/ module not on the canonical importable path.
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
    logger.info("[split=%s] looking up demographics for %d users", split, len(unique_users))
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


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce categorical key columns + float32 E so two frames are comparable."""
    df = df.copy()
    for c in JOIN_KEYS:
        # Both the on-disk parquet and the producer wire categories; coerce
        # to plain string for stable joins (category orderings can differ).
        df[c] = df[c].astype(str)
    df["E_per_user"] = df["E_per_user"].astype(np.float32)
    return df


def _summarise(df: pd.DataFrame, label: str) -> None:
    logger.info(
        "%s: %d rows | scenarios=%s | subgroup_attrs=%s | channels=%s",
        label,
        len(df),
        sorted(df["scenario"].unique().tolist()),
        sorted(df["subgroup_attr"].unique().tolist()),
        len(df["channel"].unique()),
    )


def parity_check(new: pd.DataFrame, gt: pd.DataFrame) -> int:
    """Return exit code: 0 on parity, non-zero on any mismatch."""
    new = _normalise(new)
    gt = _normalise(gt)

    _summarise(new, "new (producer)")
    _summarise(gt, "ground truth (paper/)")

    merged = new.merge(
        gt,
        on=JOIN_KEYS,
        how="outer",
        suffixes=("_new", "_gt"),
        indicator=True,
    )

    only_new = merged[merged["_merge"] == "left_only"]
    only_gt = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    logger.info(
        "merged: both=%d  only_new=%d  only_gt=%d  total=%d",
        len(both),
        len(only_new),
        len(only_gt),
        len(merged),
    )

    failed = False

    if len(only_new):
        failed = True
        logger.error("FAIL: %d rows present in producer output but not in ground truth", len(only_new))
        logger.error("first 5 rows:\n%s", only_new.head().to_string())
    if len(only_gt):
        failed = True
        logger.error("FAIL: %d rows present in ground truth but not in producer output", len(only_gt))
        logger.error("first 5 rows:\n%s", only_gt.head().to_string())

    # Numerical parity (both side only — the row-set check above caught the rest).
    e_new = both["E_per_user_new"].to_numpy()
    e_gt = both["E_per_user_gt"].to_numpy()
    if not np.array_equal(e_new, e_gt, equal_nan=True):
        failed = True
        diff = both[~(
            (np.isnan(e_new) & np.isnan(e_gt)) | (e_new == e_gt)
        )].copy()
        logger.error(
            "FAIL: E_per_user mismatch on %d / %d both-rows", len(diff), len(both)
        )
        diff = diff.assign(
            abs_diff=np.abs(diff["E_per_user_new"].to_numpy() - diff["E_per_user_gt"].to_numpy())
        )
        worst = diff.nlargest(10, "abs_diff")
        logger.error("worst 10 by abs_diff:\n%s", worst.to_string())
    else:
        logger.info("PASS: E_per_user is byte-equal on all %d both-rows", len(both))

    if failed:
        logger.error("§1 LOCF parity FAILED — STOP. See above for details.")
        return 1
    logger.info("§1 LOCF parity PASSED ✓")
    return 0


def main() -> int:
    """Run the LOCF per-user error parity checker CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locf-pairs", type=Path, default=DEFAULT_LOCF_PAIRS)
    p.add_argument("--gt-parquet", type=Path, default=DEFAULT_GT_PARQUET)
    p.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--age-bins", type=int, nargs="+", default=DEFAULT_AGE_BINS)
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing the verified parquet (in-memory parity check only)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Refuse to clobber paper/.
    if "paper-verification" not in args.out_parquet.parts and "paper" in args.out_parquet.parts:
        logger.error(
            "REFUSING to write to %s — output path must be under paper-verification/",
            args.out_parquet,
        )
        return 2

    if not args.locf_pairs.exists():
        logger.error("LOCF pairs dir does not exist: %s", args.locf_pairs)
        return 2
    if not args.gt_parquet.exists():
        logger.error("Ground-truth parquet does not exist: %s", args.gt_parquet)
        return 2

    # --- 1. Build subgroup_mapping (per split) -------------------------------
    subgroup_mappings: dict[str, dict[int, dict[str, str]]] = {}
    for split in args.splits:
        sg = _build_subgroup_mapping(args.locf_pairs, split, args.age_bins)
        if sg is None:
            logger.error("Could not build subgroup mapping for split=%s", split)
            return 2
        subgroup_mappings[split] = sg
        logger.info("[split=%s] subgroup mapping: %d samples", split, len(sg))

    # --- 2. Run the producer -------------------------------------------------
    per_user_df, _display = build_per_user_errors(
        method_pairs_dir=args.locf_pairs,
        method_name="locf",
        scenarios=args.scenarios,
        splits=args.splits,
        subgroup_mappings=subgroup_mappings,
        include_auc=True,
        exclude_unknown=False,
    )
    logger.info("producer produced %d rows", len(per_user_df))
    if per_user_df.empty:
        logger.error("Producer returned an empty frame — aborting")
        return 2

    # --- 3. Write to paper-verification/ (unless --no-write) -----------------
    if not args.no_write:
        meta = {
            "method": "locf",
            "scenarios": args.scenarios,
            "splits": args.splits,
            "age_bins": args.age_bins,
            "source_pairs": str(args.locf_pairs),
            "purpose": "§1 LOCF per-user-errors parity verification",
        }
        write_per_user_errors_parquet(per_user_df, args.out_parquet, meta=meta)
        logger.info("wrote %s", args.out_parquet)

    # --- 4. Parity check vs ground truth -------------------------------------
    logger.info("loading ground truth: %s", args.gt_parquet)
    gt = pd.read_parquet(args.gt_parquet)
    gt = gt[gt["method"] == "locf"][PER_USER_ERRORS_PARQUET_COLUMNS].copy()
    logger.info("ground truth (locf slice): %d rows", len(gt))

    return parity_check(per_user_df[PER_USER_ERRORS_PARQUET_COLUMNS].copy(), gt)


if __name__ == "__main__":
    sys.exit(main())
