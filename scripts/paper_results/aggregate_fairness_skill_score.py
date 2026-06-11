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

Usage::

    PYTHONPATH=src python scripts/paper_results/aggregate_fairness_skill_score.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output results/paper/fairness_skill_score_bootstrap.csv \
        --baseline-method linear
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from downstream_evaluation.evaluation.bootstrap_skill_rank import POINT_DRAW, read_draws_parquet

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
    p.add_argument("--clip-lower", type=float, default=1e-2)
    p.add_argument("--clip-upper", type=float, default=100.0)
    p.add_argument("--ci-level", type=float, default=0.95)
    args = p.parse_args()

    draws, _ = read_draws_parquet(args.draws)
    sub = draws[draws["subgroup_attr"].isin(SENSITIVE_ATTRS)]
    if sub.empty:
        log.warning("No subgroup rows in draws — writing empty fairness skill score CSV.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(columns=["method", "scope", "point", "se", "ci_lo", "ci_hi", "n_boot"]).to_csv(
            args.output,
            index=False,
        )
        return 0

    methods = sorted(draws["method"].unique())
    base = args.baseline_method
    n_boot = int(draws.loc[draws["draw"] != POINT_DRAW, "draw"].nunique())
    # S^(G)_j per draw: sg[method][attr][draw] = score
    sg: dict[str, dict[str, dict[int, float]]] = {
        m: {a: {} for a in SENSITIVE_ATTRS} for m in methods
    }

    for (attr, b), g in sub.groupby(["subgroup_attr", "draw"]):
        # disparity D_{r} per (task, method) = max−min of E across subgroup values
        disp = g.groupby(["task", "method"])["E"].agg(lambda x: x.max() - x.min()).unstack("method")
        if base not in disp.columns:
            continue
        d_base = disp[base]
        for m in methods:
            if m not in disp.columns:
                continue
            d_model = disp[m]
            mask = (d_base > 0) & np.isfinite(d_base) & np.isfinite(d_model)
            if mask.sum() == 0:
                continue
            ratios = np.clip(
                (d_model[mask] / d_base[mask]).to_numpy(), args.clip_lower, args.clip_upper
            )
            sg[m][attr][int(b)] = 1.0 - float(np.exp(np.mean(np.log(ratios))))

    rows = []
    for m in methods:
        for attr in SENSITIVE_ATTRS:
            pt = sg[m][attr].get(POINT_DRAW)
            boot = [v for b, v in sg[m][attr].items() if b != POINT_DRAW]
            rows.append(
                {
                    "method": m,
                    "scope": attr,
                    **_summarise(boot, args.ci_level, pt),
                    "n_boot": n_boot,
                }
            )
        # Macro over the sensitive attributes, per draw; the point draw is the value.
        boot_draws = sorted({b for a in SENSITIVE_ATTRS for b in sg[m][a] if b != POINT_DRAW})
        macro_boot = []
        for b in boot_draws:
            vals = [sg[m][a][b] for a in SENSITIVE_ATTRS if b in sg[m][a]]
            if vals:
                macro_boot.append(float(np.mean(vals)))
        pt_vals = [sg[m][a][POINT_DRAW] for a in SENSITIVE_ATTRS if POINT_DRAW in sg[m][a]]
        macro_point = float(np.mean(pt_vals)) if pt_vals else None
        rows.append(
            {
                "method": m,
                "scope": "overall",
                **_summarise(macro_boot, args.ci_level, macro_point),
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
