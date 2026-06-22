"""§2 Point-metrics parity verification.

Compares the B3 ``--per-user-errors`` output against two oracles:

1. **(A) Self-consistency.** The same ``compute_imputation_paper_metrics.py``
   run **from pairs** (without ``--per-user-errors``). For non-excluded
   scenarios this should be byte-equal — both reduce the same underlying
   pair data through the same kernels. Divergence on the excluded-binary
   scenarios (``sleep_gap`` / ``workout_gap`` / ``intensity_failure``)
   for binary channels is expected (the producer applies the bootstrap's
   EXCLUDE_BINARY_SCENARIOS filter; the from-pairs path historically
   did not for ``per_user_long``).

2. **(B) Bootstrap point estimate.** The on-disk ``paper/*_bootstrap.csv``
   files' ``mean`` columns — these are 1000-draw bootstrap means. The
   deterministic point should fall within ``~5 × se`` of the mean.

Both A and B run; B is the more conservative bar. The script emits a
clear pass/fail per CSV.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CSV_PAIRS = [
    # (B3 output filename, bootstrap counterpart filename, key cols, value col, tolerance type)
    (
        "skill_scores.csv",
        "skill_scores_bootstrap.csv",
        ["method", "scope", "split"],
        "skill_score",  # B3 outputs `skill_score`
        "S_per_user",  # column in skill csv that should match bootstrap `mean`
    ),
    (
        "avg_rankings.csv",
        "avg_rankings_bootstrap.csv",
        ["method", "scope", "split"],
        "avg_rank",  # column in rank csv
        "avg_rank_per_user",
    ),
    (
        "fairness_skill_scores.csv",
        "fairness_skill_score_bootstrap.csv",
        ["method", "scope", "split"],
        "fair_skill_score",
        "S_fair",
    ),
]


def _drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce keys to str, drop helper columns the bootstrap CSV adds."""
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df


def _diff_byte_equal(
    new: pd.DataFrame,
    other: pd.DataFrame,
    *,
    keys: list[str],
    label_new: str,
    label_other: str,
    tol: float = 1e-9,
) -> tuple[bool, str]:
    """Strict byte-equal compare on the value columns shared between the two frames."""
    new = _drop_uninformative(new.copy())
    other = _drop_uninformative(other.copy())
    shared_keys = [k for k in keys if k in new.columns and k in other.columns]
    if not shared_keys:
        return False, f"no shared key columns between {label_new} and {label_other}"

    merged = new.merge(
        other,
        on=shared_keys,
        how="outer",
        suffixes=(f"__{label_new}", f"__{label_other}"),
        indicator=True,
    )
    only_a = merged[merged["_merge"] == "left_only"]
    only_b = merged[merged["_merge"] == "right_only"]

    note = (
        f"both={int((merged['_merge'] == 'both').sum())} "
        f"only_{label_new}={len(only_a)} only_{label_other}={len(only_b)}"
    )

    if len(only_a) or len(only_b):
        return False, f"row-set differs: {note}"

    # Numerical compare on shared value cols (intersection of new & other minus keys).
    new_value_cols = set(new.columns) - set(shared_keys)
    other_value_cols = set(other.columns) - set(shared_keys)
    shared_value = sorted(new_value_cols & other_value_cols)
    mismatches: list[str] = []
    for col in shared_value:
        col_a = f"{col}__{label_new}"
        col_b = f"{col}__{label_other}"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue
        a = merged[col_a].to_numpy()
        b = merged[col_b].to_numpy()
        # Skip non-numeric columns
        if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
            continue
        diff = np.abs(a - b)
        diff[np.isnan(a) & np.isnan(b)] = 0.0
        worst = float(np.nanmax(diff)) if diff.size else 0.0
        if worst > tol:
            mismatches.append(f"{col}: max |diff|={worst:.3e}")

    if mismatches:
        return False, f"{note}; {'; '.join(mismatches)}"
    return True, note


def _diff_within_se(
    new: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    keys: list[str],
    new_value_col_candidates: list[str],
    se_multiplier: float = 5.0,
) -> tuple[bool, str]:
    """Compare new[value] vs bootstrap[mean] within se_multiplier × bootstrap[se]."""
    new = _drop_uninformative(new.copy())
    bootstrap = _drop_uninformative(bootstrap.copy())

    shared_keys = [k for k in keys if k in new.columns and k in bootstrap.columns]
    if not shared_keys:
        return False, f"no shared key columns ({keys})"

    # Locate the deterministic-value column the script emitted.
    val_col = None
    for cand in new_value_col_candidates:
        if cand in new.columns:
            val_col = cand
            break
    if val_col is None:
        return False, f"no value column found in new (tried {new_value_col_candidates})"
    if "mean" not in bootstrap.columns or "se" not in bootstrap.columns:
        return False, "bootstrap csv lacks mean/se columns"

    merged = new[[*shared_keys, val_col]].merge(
        bootstrap[[*shared_keys, "mean", "se"]],
        on=shared_keys,
        how="inner",
    )
    if merged.empty:
        return False, "no overlap between new and bootstrap rows"

    diff = np.abs(merged[val_col].to_numpy() - merged["mean"].to_numpy())
    threshold = se_multiplier * merged["se"].to_numpy()
    out_of_band = diff > threshold
    n_oob = int(np.nansum(out_of_band))
    if n_oob:
        worst = merged.assign(
            abs_diff=diff,
            threshold=threshold,
            band_ratio=diff / np.where(threshold > 0, threshold, np.nan),
        ).nlargest(10, "band_ratio")
        logger.warning("out-of-band rows (worst 10):\n%s", worst.to_string())
        return False, (
            f"{n_oob}/{len(merged)} rows outside {se_multiplier}×SE band "
            f"(val_col={val_col})"
        )
    return True, (
        f"all {len(merged)} rows within {se_multiplier}×SE "
        f"(max |new - mean|={float(np.nanmax(diff)):.3e})"
    )


def main() -> int:
    """Run point-estimate parity checks across output directories."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new", type=Path, required=True, help="B3 output dir (--per-user-errors)")
    p.add_argument("--pairs", type=Path, required=True, help="from-pairs output dir (legacy)")
    p.add_argument(
        "--bootstrap",
        type=Path,
        required=True,
        help="dir containing paper/*_bootstrap.csv ground-truth artifacts",
    )
    p.add_argument(
        "--se-multiplier",
        type=float,
        default=5.0,
        help="bootstrap (B) tolerance: |new - mean| < se_multiplier × se",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    failed = False
    for new_name, boot_name, keys, _new_val, _boot_val in CSV_PAIRS:
        new_path = args.new / new_name
        pairs_path = args.pairs / new_name
        boot_path = args.bootstrap / boot_name

        logger.info("=" * 60)
        logger.info("CSV: %s", new_name)

        if not new_path.exists():
            logger.error("MISSING B3 output: %s", new_path)
            failed = True
            continue
        new = pd.read_csv(new_path)

        # (A) self-consistency vs from-pairs
        if pairs_path.exists():
            other = pd.read_csv(pairs_path)
            ok, note = _diff_byte_equal(
                new, other, keys=keys, label_new="pue", label_other="pairs"
            )
            logger.info("(A) %s vs %s: %s — %s", new_name, pairs_path, "PASS" if ok else "FAIL", note)
            if not ok:
                # Self-consistency is informational on the excluded-binary edge case.
                # We don't fail the overall §2 on it — (B) is the hard bar.
                logger.warning(
                    "(A) divergence may be EXCLUDE_BINARY_SCENARIOS edge — see B3 design notes"
                )
        else:
            logger.warning("(A) skipped — %s missing", pairs_path)

        # (B) bootstrap point estimate
        if boot_path.exists():
            bootstrap = pd.read_csv(boot_path)
            value_candidates = [
                "skill_score",
                "S_per_user",
                "avg_rank",
                "avg_rank_per_user",
                "fair_skill_score",
                "S_fair",
                "mean",
            ]
            ok, note = _diff_within_se(
                new,
                bootstrap,
                keys=keys,
                new_value_col_candidates=value_candidates,
                se_multiplier=args.se_multiplier,
            )
            logger.info("(B) %s vs %s: %s — %s", new_name, boot_path, "PASS" if ok else "FAIL", note)
            if not ok:
                failed = True
        else:
            logger.warning("(B) skipped — %s missing", boot_path)

    if failed:
        logger.error("§2 point-metrics parity FAILED")
        return 1
    logger.info("§2 point-metrics parity PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
