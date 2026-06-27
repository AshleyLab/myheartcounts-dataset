"""Per-task skill score from existing bootstrap draws.

Skill aggregates error ratios via a within-domain geomean (see bootstrap_skill_rank.
_per_domain_skill_from_ratios). A single-task skill is the degenerate one-element
case: S_task = 1 - clip(E_method / E_baseline). This reads the per-draw errors
already written by the paper pipeline and emits per-task skill with point + SE +
percentile CI, matching the pipeline's clip bounds, baseline, and summary convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.evaluation.skill_score import (
    DEFAULT_CLIP_LOWER,
    DEFAULT_CLIP_UPPER,
)

DRAWS = Path("results/paper/bootstrap_draws.parquet")
OUT = Path("results/paper/per_task_skill_bootstrap.csv")
BASELINE = "linear"
CI_LEVEL = 0.95
POINT_DRAW = -1


def _summarise(draws: np.ndarray, point: float) -> dict[str, float]:
    """Point from the full-cohort draw; SE/CI from the resamples (matches pipeline)."""
    arr = draws[np.isfinite(draws)]
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "point": float(point),
        "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "ci_lo": float(np.percentile(arr, 100 * alpha)) if len(arr) else float("nan"),
        "ci_hi": float(np.percentile(arr, 100 * (1 - alpha))) if len(arr) else float("nan"),
    }


def main() -> None:
    df = pd.read_parquet(DRAWS)
    # Overall cohort only (skill scope = global, not demographic subgroups).
    df = df[(df["subgroup_attr"] == "all") & (df["subgroup_value"] == "all")]

    rows: list[dict] = []
    for (task, domain, task_type), g in df.groupby(["task", "domain", "task_type"]):
        wide = g.pivot(index="draw", columns="method", values="E")
        if BASELINE not in wide.columns:
            continue
        base = wide[BASELINE]
        for method in wide.columns:
            if method == BASELINE:
                continue
            ratio = (wide[method] / base).clip(DEFAULT_CLIP_LOWER, DEFAULT_CLIP_UPPER)
            skill = 1.0 - ratio
            point = float(skill.loc[POINT_DRAW]) if POINT_DRAW in skill.index else float("nan")
            resamples = skill.drop(index=POINT_DRAW, errors="ignore").to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "domain": domain,
                    "task_type": task_type,
                    **_summarise(resamples, point),
                    "n_boot": int(np.isfinite(resamples).sum()),
                }
            )

    out = pd.DataFrame(rows).sort_values(["method", "domain", "task"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows ({out['task'].nunique()} tasks x "
          f"{out['method'].nunique()} methods) -> {OUT}")


if __name__ == "__main__":
    main()
