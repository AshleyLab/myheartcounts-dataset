"""§3 Bootstrap parity verification.

Compares a fresh ``bootstrap_draws.parquet`` (produced by the Phase B
``--per-user-errors-dir`` path) against the on-disk ground truth
(``paper/bootstrap_draws.parquet``). Same seed → same boot_idx → same
draws should round-trip byte-equal on the ``E``, ``R``, ``rank`` columns.

Assertions:
  1. Row-set equality on the 8 join keys (no rows only in new, none only
     in ground truth).
  2. ``E``, ``R``, ``rank`` byte-equal after the float32 cast (parquet
     already stores them as float32).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


JOIN_KEYS = [
    "method",
    "scenario",
    "split",
    "channel",
    "channel_type",
    "subgroup_attr",
    "subgroup_value",
    "draw",
]
VALUE_COLS = ["E", "R", "rank"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in JOIN_KEYS:
        if c == "draw":
            df[c] = df[c].astype(np.int32)
        else:
            df[c] = df[c].astype(str)
    for c in VALUE_COLS:
        df[c] = df[c].astype(np.float32)
    return df


def parity_check(new: pd.DataFrame, gt: pd.DataFrame) -> int:
    new = _normalise(new)
    gt = _normalise(gt)
    logger.info("new: %d rows | gt: %d rows", len(new), len(gt))

    merged = new.merge(
        gt, on=JOIN_KEYS, how="outer", suffixes=("_new", "_gt"), indicator=True
    )
    only_new = int((merged["_merge"] == "left_only").sum())
    only_gt = int((merged["_merge"] == "right_only").sum())
    both = int((merged["_merge"] == "both").sum())
    logger.info("merged: both=%d only_new=%d only_gt=%d", both, only_new, only_gt)

    failed = False
    if only_new or only_gt:
        failed = True
        logger.error("FAIL row-set: only_new=%d only_gt=%d", only_new, only_gt)
        if only_new:
            logger.error(
                "first 5 only-new rows:\n%s",
                merged[merged["_merge"] == "left_only"].head().to_string(),
            )
        if only_gt:
            logger.error(
                "first 5 only-gt rows:\n%s",
                merged[merged["_merge"] == "right_only"].head().to_string(),
            )

    both_df = merged[merged["_merge"] == "both"]
    for col in VALUE_COLS:
        a = both_df[f"{col}_new"].to_numpy()
        b = both_df[f"{col}_gt"].to_numpy()
        if np.array_equal(a, b, equal_nan=True):
            logger.info("PASS: %s byte-equal on all %d both-rows", col, len(both_df))
            continue
        failed = True
        diff = np.abs(a - b)
        finite = np.isfinite(diff)
        n_diff = int(np.sum(diff > 0) - np.sum(np.isnan(a) & np.isnan(b)))
        worst = float(np.nanmax(diff)) if finite.any() else float("nan")
        logger.error(
            "FAIL: %s mismatch on %d rows (max |diff|=%.3e)", col, n_diff, worst
        )
        offenders = both_df.assign(abs_diff=diff).nlargest(10, "abs_diff")[
            JOIN_KEYS + [f"{col}_new", f"{col}_gt", "abs_diff"]
        ]
        logger.error("worst 10 by abs_diff:\n%s", offenders.to_string())

    if failed:
        logger.error("§3 bootstrap parity FAILED")
        return 1
    logger.info("§3 bootstrap parity PASSED ✓")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new", type=Path, required=True, help="new bootstrap_draws.parquet")
    p.add_argument("--gt", type=Path, required=True, help="ground-truth bootstrap_draws.parquet")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if not args.new.exists():
        logger.error("new parquet missing: %s", args.new)
        return 2
    if not args.gt.exists():
        logger.error("gt parquet missing: %s", args.gt)
        return 2

    new = pd.read_parquet(args.new)
    gt = pd.read_parquet(args.gt)
    return parity_check(new, gt)


if __name__ == "__main__":
    sys.exit(main())
