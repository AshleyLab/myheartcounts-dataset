#!/usr/bin/env python3
"""Reduce the HF forecasting substrate into the paper bootstrap CSVs.

The OpenMHC leaderboard HF dataset (``MyHeartCounts/OpenMHC-leaderboard-data``)
is the single source of truth.

* **Skill + rank** are reduced directly from the pre-computed per-draw reference
  ``forecasting/bootstrap/draws.parquet`` (long frame
  ``(reduction, model, scope, metric, draw, value)``) via the same ``_summarize``
  helper the pipeline uses — this avoids the ``bootstrap_skill_rank`` rank kernel,
  which requires pandas >= 2.1 (``DataFrame.stack(future_stack=...)``).
* **Fairness** (deterministic point + BCa) is computed by
  ``bootstrap_fair_skill_score`` over the per-method substrate
  (``forecasting/<model>.parquet``), exactly as ``run_paper_pipeline.py`` does —
  the BCa acceleration needs the leave-one-user-out jackknife, which is not in the
  draws reference.

Writes the three CSVs the arXiv forecasting table generator consumes:
    forecasting_skill_score_bootstrap.csv
    forecasting_grouped_metric_rank_bootstrap.csv
    forecasting_fairness_skill_score_bootstrap.csv

Demographics for the fairness BCa are resolved via the labels API; set
``MHC_DATA_DIR`` (e.g. ``~/.cache/openmhc/data-full``) so
``<root>/labels/{last_labels,enrollment_info}.json`` resolve.

Usage:
    MHC_DATA_DIR=~/.cache/openmhc/data-full python \
        scripts/paper_results/forecasting/build_forecasting_bootstrap_from_hf.py \
        --out-dir /tmp/forecasting_hf_csvs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402
from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (  # noqa: E402
    bootstrap_fair_skill_score,
)
from forecasting_evaluation.metrics.bootstrap_skill_rank import _summarize  # noqa: E402
from forecasting_evaluation.metrics.per_user_errors import (  # noqa: E402
    PER_USER_METRICS_PARQUET_COLUMNS,
)

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"
DRAWS_PATH = "forecasting/bootstrap/draws.parquet"
N_BOOT, SEED, CI_LEVEL = 1000, 42, 0.95
AGE_BINS = (18, 30, 40, 50, 60)


def _reduce_draws(draws: pd.DataFrame, reduction: str, key_cols: list[str]) -> pd.DataFrame:
    """Reduce per-draw values to {mean, se, ci_lo, ci_hi, n_boot} per key.

    Mirrors the pipeline's percentile-CI reduction (``_summarize``) so the CSVs
    are interchangeable with a from-substrate ``bootstrap_skill_rank`` run.
    """
    sub = draws[draws["reduction"] == reduction]
    rows = []
    for key, grp in sub.groupby(key_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(key_cols, (str(k) for k in key)))
        rec.update(_summarize(grp["value"].to_numpy(), CI_LEVEL))
        rows.append(rec)
    return pd.DataFrame(rows)
# The 10 paper models (matches forecasting/bootstrap/draws.meta.json).
MODELS = [
    "seasonal_naive",
    "autoARIMA",
    "autoETS",
    "mixlinear",
    "dlinear",
    "segrnn",
    "toto_zeroshot_ctx4096",
    "toto_finetuned_ctx4096",
    "chronos2_zeroshot",
    "chronos2_finetuned",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = []
    for name in MODELS:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=f"forecasting/{name}.parquet",
            repo_type="dataset",
            revision=args.revision,
        )
        frames.append(pd.read_parquet(path)[PER_USER_METRICS_PARQUET_COLUMNS])
    substrate = pd.concat(frames, ignore_index=True)
    print(f"substrate: {len(substrate)} rows, {substrate['user_id'].nunique()} users "
          f"({time.time() - t0:.0f}s)")

    models = {name: {"path": "", "display_name": name} for name in MODELS}

    # --- Skill + rank: reduce directly from the pre-computed HF draws reference ---
    draws_path = hf_hub_download(
        repo_id=args.repo_id, filename=DRAWS_PATH, repo_type="dataset", revision=args.revision
    )
    draws = pd.read_parquet(draws_path)
    _reduce_draws(draws, "skill", ["model", "scope"]).to_csv(
        args.out_dir / "forecasting_skill_score_bootstrap.csv", index=False
    )
    _reduce_draws(draws, "rank", ["model", "scope", "metric"]).to_csv(
        args.out_dir / "forecasting_grouped_metric_rank_bootstrap.csv", index=False
    )
    print(f"skill/rank reduced from draws ({time.time() - t0:.0f}s)")

    from labels.api import ENROLLMENT_PATH, LABELS_PATH

    if not LABELS_PATH or not ENROLLMENT_PATH:
        raise SystemExit("labels/enrollment unresolved — set MHC_DATA_DIR.")
    ftables = bootstrap_fair_skill_score(
        models=models,
        baseline_model=_spec.PAPER_BASELINE,
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        labels_path=str(LABELS_PATH),
        enrollment_path=str(ENROLLMENT_PATH),
        age_bins=AGE_BINS,
        n_boot=N_BOOT,
        seed=SEED,
        ci_level=CI_LEVEL,
        within_user_aggregation="micro",
        bca=True,
        per_user_metrics=substrate,
    )
    ftables["fairness_skill_scores_point"].to_csv(
        args.out_dir / "forecasting_fairness_skill_score.csv", index=False
    )
    ftables["fairness_skill_scores"].to_csv(
        args.out_dir / "forecasting_fairness_skill_score_bootstrap.csv", index=False
    )
    print(f"fairness done ({time.time() - t0:.0f}s). Wrote CSVs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
