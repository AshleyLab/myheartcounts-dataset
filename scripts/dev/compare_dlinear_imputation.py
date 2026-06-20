"""Compare DLinear imputation metrics between public and private runs.

Pure JSON-to-JSON: reads each side's metrics file and walks the shared
``scenarios -> split -> {continuous, binary, per_channel}`` shape, printing
deltas with a PARITY OK / MISMATCH banner at ``atol=1e-4``.

The public ``mhc-impute-eval`` writes inline metrics into
``results.json`` (when ``evaluation.compute_metrics=true``). The private
codebase used ``compute_metrics=false, save_pairs=true`` and computed
metrics offline into ``pairs/aggregated_metrics.json``. Both files share
the same schema produced by ``ScenarioMetricsAccumulator.compute()`` in
``src/imputation_evaluation/evaluation/evaluator.py``.

Usage:
    python scripts/dev/compare_dlinear_imputation.py \
        --public  results/imputation_eval/<run>/results.json \
        --private "$HOME/MHC-benchmark/results/imputation_eval/dlinear_max91d_<timestamp>/pairs/aggregated_metrics.json"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_ATOL = 1e-4
_atol = DEFAULT_ATOL

# Aggregate metrics to compare per (scenario, split).
CONT_KEYS = ("mean_normalized_rmse", "mean_normalized_mse", "mean_normalized_mae")
BIN_KEYS = ("macro_balanced_accuracy", "macro_roc_auc")
# Per-channel metrics to compare. Continuous channels carry rmse/mse/mae +
# their normalized variants; binary channels carry balanced_accuracy/roc_auc.
PER_CH_CONT_KEYS = ("normalized_rmse", "normalized_mse", "normalized_mae")
PER_CH_BIN_KEYS = ("balanced_accuracy", "roc_auc")


def _is_close(a: float, b: float, atol: float | None = None) -> bool:
    """Treat NaN==NaN as match; otherwise abs(a-b)<=atol."""
    if atol is None:
        atol = _atol
    a_nan = isinstance(a, float) and math.isnan(a)
    b_nan = isinstance(b, float) and math.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return abs(a - b) <= atol


def _diff(a: float, b: float) -> float:
    a_nan = isinstance(a, float) and math.isnan(a)
    b_nan = isinstance(b, float) and math.isnan(b)
    if a_nan or b_nan:
        return float("nan")
    return abs(a - b)


def _get(d: dict, key: str, default: float = float("nan")) -> float:
    v = d.get(key, default)
    return float(v) if v is not None else float("nan")


def _walk(pub: dict, priv: dict) -> tuple[int, int, float]:
    """Print per-(scenario,split,group,metric) diffs. Returns (n_match, n_total, max_abs_diff)."""
    pub_scen = pub.get("scenarios", {})
    priv_scen = priv.get("scenarios", {})

    shared = sorted(set(pub_scen) & set(priv_scen))
    pub_only = sorted(set(pub_scen) - set(priv_scen))
    priv_only = sorted(set(priv_scen) - set(pub_scen))
    if pub_only or priv_only:
        print(f"  WARNING: scenario mismatch — public-only={pub_only}, private-only={priv_only}")

    n_match = 0
    n_total = 0
    max_diff = 0.0

    for scen in shared:
        for split in ("val", "test"):
            pub_split = pub_scen[scen].get(split)
            priv_split = priv_scen[scen].get(split)
            if pub_split is None or priv_split is None:
                continue
            # Skip if public is in pairs-only mode (no inline metrics).
            if pub_split.get("pairs_only"):
                print(
                    f"  [{scen}/{split}] public is pairs_only — re-run with "
                    f"evaluation.compute_metrics=true or use offline aggregation"
                )
                continue

            # Continuous aggregates
            pub_c = pub_split.get("continuous", {})
            priv_c = priv_split.get("continuous", {})
            for key in CONT_KEYS:
                pv, rv = _get(pub_c, key), _get(priv_c, key)
                ok = _is_close(pv, rv)
                d = _diff(pv, rv)
                if not math.isnan(d):
                    max_diff = max(max_diff, d)
                n_total += 1
                n_match += int(ok)
                status = "OK" if ok else "MISMATCH"
                print(
                    f"  [{scen}/{split}/continuous] {key:<22s} "
                    f"pub={pv:.6f}  priv={rv:.6f}  diff={d:.2e}  {status}"
                )

            # Binary aggregates
            pub_b = pub_split.get("binary", {})
            priv_b = priv_split.get("binary", {})
            for key in BIN_KEYS:
                pv, rv = _get(pub_b, key), _get(priv_b, key)
                ok = _is_close(pv, rv)
                d = _diff(pv, rv)
                if not math.isnan(d):
                    max_diff = max(max_diff, d)
                n_total += 1
                n_match += int(ok)
                status = "OK" if ok else "MISMATCH"
                print(
                    f"  [{scen}/{split}/binary]     {key:<22s} "
                    f"pub={pv:.6f}  priv={rv:.6f}  diff={d:.2e}  {status}"
                )

            # Per-channel
            pub_pc = pub_split.get("per_channel", {})
            priv_pc = priv_split.get("per_channel", {})
            for ch_key in sorted(set(pub_pc) & set(priv_pc), key=lambda s: int(s.split("_")[1])):
                pub_ch = pub_pc[ch_key]
                priv_ch = priv_pc[ch_key]
                # Detect continuous vs binary by which keys are present.
                if "normalized_rmse" in priv_ch:
                    keys = PER_CH_CONT_KEYS
                    group = "cont"
                else:
                    keys = PER_CH_BIN_KEYS
                    group = "bin"
                for key in keys:
                    pv, rv = _get(pub_ch, key), _get(priv_ch, key)
                    ok = _is_close(pv, rv)
                    d = _diff(pv, rv)
                    if not math.isnan(d):
                        max_diff = max(max_diff, d)
                    n_total += 1
                    n_match += int(ok)
                    if not ok:
                        print(
                            f"  [{scen}/{split}/{group}/{ch_key}] {key:<18s} "
                            f"pub={pv:.6f}  priv={rv:.6f}  diff={d:.2e}  MISMATCH"
                        )

    return n_match, n_total, max_diff


def main() -> int:
    """Compare public vs. private imputation metrics and report parity.

    Returns:
        Process exit code: ``0`` if all metric values match within tolerance,
        otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True, help="public results.json")
    parser.add_argument(
        "--private",
        type=Path,
        required=True,
        help="private aggregated_metrics.json (or results.json if it carries inline metrics)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"abs tolerance (default {DEFAULT_ATOL})",
    )
    args = parser.parse_args()

    global _atol  # noqa: PLW0603
    _atol = args.atol

    print(f"Public:  {args.public}")
    print(f"Private: {args.private}")
    print(f"Tolerance: atol={_atol:.0e}")
    print()

    pub = json.loads(args.public.read_text())
    priv = json.loads(args.private.read_text())

    n_match, n_total, max_diff = _walk(pub, priv)

    print()
    print(f"Compared {n_total} metric values; {n_match} within tolerance.")
    print(f"Max abs diff: {max_diff:.2e}")
    if n_match == n_total:
        print("RESULT: PARITY CONFIRMED")
        return 0
    print(f"RESULT: {n_total - n_match} MISMATCH(es) — see lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
