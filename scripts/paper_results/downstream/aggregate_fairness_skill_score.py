#!/usr/bin/env python
r"""Fairness Skill Score reducer (phase-2 sidecar) for the downstream track.

Reads ``bootstrap_draws.parquet`` (phase 1) and emits
``fairness_skill_score_bootstrap.csv``: per-method fairness skill score for each
sensitive attribute (age_group, sex) plus the macro-averaged overall score, all
with mean / SE / percentile CI across draws.

Formulation (disparity-ratio vs the baseline, mirrors the skill-score machinery):

    For each task r and attribute G:
        D_{r,j} = max_g E_{r,j}^{(g)} − min_g E_{r,j}^{(g)}   (model j's subgroup spread)
        D_{r,b} = same for the baseline; drop r when D_{r,b} ≤ 0 or NaN.
        ratio_r = clip(D_{r,j} / D_{r,b}, ℓ, u)
    Per attribute:  S^{(G)}_j = 1 − GeometricMean_r(ratio_r)
    Macro-average:  S_fair_j  = mean over attributes of S^{(G)}_j

Per-draw D_j and D_b share the same resampled cohort (the ``draw`` axis), so the
pairing is preserved; summary statistics aggregate across draws.

The reported value is the deterministic point estimate (full cohort). S is a
geometric mean of clipped disparity ratios, so it is right-skewed and its
bootstrap mean sits below the point — the plain percentile CI is therefore biased
low. When per-user predictions are supplied (``--predictions_dir``/``--csvs_dir``)
the reducer also emits a BCa (bias-corrected & accelerated) interval
(``bca_lo``/``bca_hi``) that re-anchors the interval near the point and corrects
for that bias and skew; its acceleration term is the exact leave-one-user-out
jackknife of S. Without predictions the BCa columns are NaN (percentile CI only).

Usage::

    PYTHONPATH=src python scripts/paper_results/downstream/aggregate_fairness_skill_score.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output results/paper/fairness_skill_score_bootstrap.csv \
        --baseline-method linear \
        --predictions_dir results/eval/final/predictions --csvs_dir results/eval/final
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    POINT_DRAW,
    _attr_disparity_ratio_skill,
    _bca_interval,
    align_across_methods,
    jackknife_fairness_skill,
    load_method_predictions,
    load_subgroup_map,
    read_draws_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SENSITIVE_ATTRS = ("age_group", "sex")


def _summarise(values: list[float], ci_level: float, point: float | None = None) -> dict[str, float]:
    """Point estimate (full cohort) as the value + SE / percentile-CI from the draws."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    center = (
        float(point)
        if point is not None and np.isfinite(point)
        else (float(np.mean(arr)) if len(arr) else float("nan"))
    )
    if len(arr) == 0:
        return {"point": center, "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    a = (1.0 - ci_level) / 2.0
    return {
        "point": center,
        "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "ci_lo": float(np.percentile(arr, 100 * a)),
        "ci_hi": float(np.percentile(arr, 100 * (1 - a))),
    }


def _warn_on_point_drift(
    jack_point: dict[tuple[str, str], float],
    point_by_key: dict[tuple[str, str], float | None],
    tol: float = 1e-3,
) -> None:
    """Warn if the full-cohort jackknife point diverges from the draws POINT_DRAW.

    The two are the same statistic on the same cohort, so they agree up to the
    draws parquet's float32 round-trip (~1e-6). A drift well above that (the
    tolerance is set to catch ~1e-2 differences) means the predictions and
    bootstrap_draws.parquet were produced by different eval runs, which would put
    the BCa acceleration and the bootstrap draws on inconsistent cohorts.
    """
    worst, worst_key = 0.0, None
    for key, jp in jack_point.items():
        pt = point_by_key.get(key)
        if pt is None or not np.isfinite(pt) or not np.isfinite(jp):
            continue
        drift = abs(float(jp) - float(pt))
        if drift > worst:
            worst, worst_key = drift, key
    if worst > tol:
        log.warning(
            "Jackknife point disagrees with the draws POINT_DRAW by up to %.3g (at %s) — "
            "predictions and bootstrap_draws.parquet look like different runs; the BCa "
            "interval and the percentile CI would be inconsistent.",
            worst,
            worst_key,
        )


def main() -> int:
    """Compute per-attribute + macro fairness skill scores from the draws parquet."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--draws", type=Path, required=True, help="bootstrap_draws.parquet from phase 1")
    p.add_argument(
        "--output", type=Path, required=True, help="Output CSV (fairness_skill_score_bootstrap.csv)"
    )
    p.add_argument("--baseline-method", default="linear")
    p.add_argument(
        "--disparity-mode",
        choices=("maxmin", "mean_pairwise"),
        default="maxmin",
        help="Subgroup disparity D: 'maxmin' (worst-case max−min = max pairwise |ΔE|, "
        "default/current) or 'mean_pairwise' (mean unordered-pairwise |ΔE| — the Gini "
        "mean difference). Threaded into both the draws path and the BCa jackknife so "
        "they stay consistent.",
    )
    p.add_argument("--clip-lower", type=float, default=1e-2)
    p.add_argument("--clip-upper", type=float, default=100.0)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument(
        "--predictions_dir",
        type=Path,
        default=None,
        help="Per-(method, task) test.parquet root. When set, also emit a BCa CI "
        "(bca_lo/bca_hi) from the leave-one-user-out jackknife.",
    )
    p.add_argument(
        "--csvs_dir",
        type=Path,
        default=None,
        help="Dir with eval_*.csv for task_type lookup. Required with --predictions_dir.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to load for the jackknife (default: every method in the draws).",
    )
    args = p.parse_args()
    if args.predictions_dir is not None and args.csvs_dir is None:
        p.error("--csvs_dir is required when --predictions_dir is given")

    draws, _ = read_draws_parquet(args.draws)
    sub = draws[draws["subgroup_attr"].isin(SENSITIVE_ATTRS)]
    if sub.empty:
        log.warning("No subgroup rows in draws — writing empty fairness skill score CSV.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        cols = ["method", "scope", "point", "se", "ci_lo", "ci_hi", "bca_lo", "bca_hi", "n_boot"]
        pd.DataFrame(columns=cols).to_csv(args.output, index=False)
        return 0

    methods = sorted(draws["method"].unique())
    base = args.baseline_method
    n_boot = int(draws.loc[draws["draw"] != POINT_DRAW, "draw"].nunique())
    # S^(G)_j per draw: sg[method][attr][draw] = score
    sg: dict[str, dict[str, dict[int, float]]] = {
        m: {a: {} for a in SENSITIVE_ATTRS} for m in methods
    }

    for (attr, b), g in sub.groupby(["subgroup_attr", "draw"]):
        # Disparity-ratio fairness skill per (task, method), reduced via the same
        # function the BCa jackknife uses so the draws path and the jackknife stay
        # consistent by construction (and the disparity mode applies to both).
        scores = _attr_disparity_ratio_skill(
            g, methods, base, args.clip_lower, args.clip_upper, disparity=args.disparity_mode
        )
        for m, s in scores.items():
            sg[m][attr][int(b)] = s

    # Bootstrap draws + point estimate per (method, scope); scope = each sensitive
    # attribute plus the macro-average "overall".
    scopes = (*SENSITIVE_ATTRS, "overall")
    boot_by_key: dict[tuple[str, str], list[float]] = {}
    point_by_key: dict[tuple[str, str], float | None] = {}
    for m in methods:
        for attr in SENSITIVE_ATTRS:
            point_by_key[(m, attr)] = sg[m][attr].get(POINT_DRAW)
            boot_by_key[(m, attr)] = [v for b, v in sg[m][attr].items() if b != POINT_DRAW]
        # Macro over the sensitive attributes, per draw; the point draw is the value.
        boot_draws = sorted({b for a in SENSITIVE_ATTRS for b in sg[m][a] if b != POINT_DRAW})
        macro_boot = []
        for b in boot_draws:
            vals = [sg[m][a][b] for a in SENSITIVE_ATTRS if b in sg[m][a]]
            if vals:
                macro_boot.append(float(np.mean(vals)))
        pt_vals = [sg[m][a][POINT_DRAW] for a in SENSITIVE_ATTRS if POINT_DRAW in sg[m][a]]
        point_by_key[(m, "overall")] = float(np.mean(pt_vals)) if pt_vals else None
        boot_by_key[(m, "overall")] = macro_boot

    # Leave-one-user-out jackknife (for the BCa acceleration) needs per-user
    # predictions, so it only runs when --predictions_dir is supplied.
    jack_by_key: dict[tuple[str, str], np.ndarray] = {}
    if args.predictions_dir is not None:
        load_methods = args.methods or methods
        aligned = align_across_methods(
            {mm: load_method_predictions(args.predictions_dir, mm, args.csvs_dir) for mm in load_methods}
        )
        subgroup_map = load_subgroup_map(args.predictions_dir)
        if subgroup_map is None:
            log.warning(
                "No subgroup map under %s — emitting percentile CI only (BCa columns NaN).",
                args.predictions_dir,
            )
        else:
            jack_by_key, jack_point = jackknife_fairness_skill(
                aligned,
                subgroup_map,
                SENSITIVE_ATTRS,
                base,
                clip_lower=args.clip_lower,
                clip_upper=args.clip_upper,
                disparity=args.disparity_mode,
            )
            _warn_on_point_drift(jack_point, point_by_key)

    rows = []
    for m in methods:
        for scope in scopes:
            pt = point_by_key.get((m, scope))
            boot = boot_by_key.get((m, scope), [])
            bca_lo, bca_hi = float("nan"), float("nan")
            jack = jack_by_key.get((m, scope))
            if jack is not None and pt is not None and np.isfinite(pt):
                bca_lo, bca_hi = _bca_interval(
                    np.asarray(boot, dtype=np.float64), float(pt), jack, args.ci_level
                )
            rows.append(
                {
                    "method": m,
                    "scope": scope,
                    **_summarise(boot, args.ci_level, pt),
                    "bca_lo": bca_lo,
                    "bca_hi": bca_hi,
                    "n_boot": n_boot,
                }
            )

    import pandas as pd

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
    log.info("Wrote %s (%d rows)", args.output, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
