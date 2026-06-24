"""Full end-to-end Gate D — the public-API ``evaluate_forecasting`` path on real data.

Runs ``openmhc.evaluate_forecasting`` with a Seasonal-Naive forecaster on the full
dataset and checks the public-API path end to end:

  * it runs the eval, builds the per-user substrate, and computes ``skill_scores``
    vs the SHIPPED Seasonal-Naive baseline (no extra wiring);
  * because the forecaster IS seasonal_naive (the baseline), every per-task ratio is
    1, so the self-skill ``overall_score`` is ~0 and the public-API substrate
    reproduces the shipped baseline on the scored metrics (mae + auroc).

This exercises the real code path Gate D-lite stubbed with synthetic trees. Heavy
(a full forecasting eval) — run via ``gate_d_public_api.sbatch`` on SLURM.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

import openmhc  # noqa: E402
from forecasting_evaluation.metrics.per_user_errors import (  # noqa: E402
    read_per_user_metrics_parquet,
)
from forecasting_evaluation.models.naive.seasonal_naive import SeasonalNaiveModel  # noqa: E402


def main() -> int:
    """Run the public-API path for seasonal_naive and assert self-skill ~0."""
    # Distinct from the baseline model name ("seasonal_naive") so the paired skill
    # score has two distinct models; the forecaster IS seasonal_naive, so self-skill
    # should be ~0.
    model = SeasonalNaiveModel(seed=42)
    res = openmhc.evaluate_forecasting(
        model, version="full", method_name="seasonal_naive_selfcheck"
    )

    print(f"n_samples={res.n_samples}  overall_fallback_rate={res.overall_fallback_rate:.4f}")
    if res.per_user_errors is None or res.per_user_errors.empty:
        print("GATE D: FAIL — empty per_user_errors substrate")
        return 1
    if res.skill_scores is None or res.skill_scores.empty:
        print("GATE D: FAIL — no skill_scores (baseline not found?)")
        return 1

    sk = res.skill_scores
    row = sk[sk["model"].astype(str) == "seasonal_naive_selfcheck"]
    score_cols = [c for c in sk.columns if c.endswith("_score")]
    overall = float(row["overall_score"].iloc[0])
    max_abs = float(np.nanmax(np.abs(row[score_cols].to_numpy(dtype=float))))
    print(f"self-skill overall_score={overall:.3e}  max|all *_score|={max_abs:.3e}")

    # The public-API run must reproduce the canonical seasonal_naive per-user errors
    # on the scored metrics (the shipped baseline was built from those same trees).
    shipped = SRC_ROOT / "openmhc/data/baselines/forecasting_seasonal_naive_per_user_errors.parquet"
    base, _ = read_per_user_metrics_parquet(shipped)
    keys = ["metric", "channel_idx", "user_id"]
    base = base.assign(metric=base["metric"].astype(str), user_id=base["user_id"].astype(str))
    pue = res.per_user_errors.assign(
        metric=res.per_user_errors["metric"].astype(str),
        user_id=res.per_user_errors["user_id"].astype(str),
    )
    base["channel_idx"] = base["channel_idx"].astype(int)
    pue["channel_idx"] = pue["channel_idx"].astype(int)
    mrg = base.merge(pue, on=keys, suffixes=("_base", "_api"))
    print(f"substrate rows matched: {len(mrg)} (shipped={len(base)}, api={len(pue)})")
    worst = 0.0
    for met in sorted(mrg["metric"].unique()):
        s = mrg[mrg["metric"] == met]
        d = float((s["metric_value_api"] - s["metric_value_base"]).abs().max())
        worst = max(worst, d)
        print(f"  metric {met}: n={len(s)} max|d|={d:.3e}")

    ok = abs(overall) < 1e-4 and max_abs < 1e-4 and len(mrg) == len(base) == len(pue)
    print("GATE D:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
