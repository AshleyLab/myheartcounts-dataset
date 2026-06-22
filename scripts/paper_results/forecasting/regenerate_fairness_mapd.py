"""Regenerate the canonical forecasting fairness CSVs under the MAPD disparity.

The fairness per-task disparity switched from max-min to MAPD (mean absolute
pairwise difference). For ``sex`` (|G|=2) this is identical to max-min, but for
``age_group`` (up to 5 buckets) it differs, so the fairness leaderboard moves
(verified: sex unchanged, age_group/overall shift modestly). Skill + rank are
unaffected by this change (proven by the substrate parity gate), so only the two
fairness CSVs are regenerated:

    <summary-dir>/forecasting_fairness_skill_score.csv            (deterministic point)
    <summary-dir>/forecasting_fairness_skill_score_bootstrap.csv  (1000-draw BCa CIs)

The previous max-min CSVs are backed up to ``*.maxmin.bak`` first. Params match
the canonical run (configs/paper/sweep_forecasting.yaml): seasonal_naive baseline,
mae + auroc, micro/user, n_boot=1000, seed=42, BCa on, age_bins 18/30/40/50/60.

Heavy (builds the 10-model substrate + a 1000-draw fairness bootstrap) — run via
``regenerate_fairness_mapd.sbatch`` on SLURM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402
from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (  # noqa: E402
    bootstrap_fair_skill_score,
)
from forecasting_evaluation.metrics.per_user_errors import build_per_user_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("regenerate_fairness_mapd")

N_BOOT, SEED, CI_LEVEL = 1000, 42, 0.95
AGE_BINS = (18, 30, 40, 50, 60)


def main() -> int:
    """Rebuild the substrate, recompute MAPD fairness, and replace the canonical CSVs."""
    p = argparse.ArgumentParser(description="Regenerate canonical fairness CSVs under MAPD.")
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "results/forecasting_eval/simurgh/summary/forecasting_bca_20260618",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute + diff but do not overwrite CSVs.")
    args = p.parse_args()

    models = json.loads((args.summary_dir / "skill_rank_models.json").read_text())
    models = models.get("models", models)
    models = {
        name: {
            "path": str(path if Path(path).is_absolute() else REPO_ROOT / path),
            "display_name": name,
        }
        for name, path in models.items()
    }
    logger.info("Building substrate for %d models from %s", len(models), args.summary_dir)

    substrate = build_per_user_metrics(
        models=models,
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
    )
    logger.info("Substrate: %d rows. Running MAPD fairness (point + %d-draw BCa)...", len(substrate), N_BOOT)

    from labels.api import ENROLLMENT_PATH, LABELS_PATH

    if not LABELS_PATH or not ENROLLMENT_PATH:
        raise SystemExit("labels/enrollment unresolved — set MHC_DATA_DIR.")

    tables = bootstrap_fair_skill_score(
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

    point_path = args.summary_dir / "forecasting_fairness_skill_score.csv"
    boot_path = args.summary_dir / "forecasting_fairness_skill_score_bootstrap.csv"
    if args.dry_run:
        logger.info("[dry-run] point rows=%d, bootstrap rows=%d (not written)",
                    len(tables["fairness_skill_scores_point"]), len(tables["fairness_skill_scores"]))
        return 0

    for path in (point_path, boot_path):
        if path.exists():
            backup = path.with_suffix(".csv.maxmin.bak")
            path.replace(backup)
            logger.info("Backed up %s -> %s", path.name, backup.name)
    tables["fairness_skill_scores_point"].to_csv(point_path, index=False)
    tables["fairness_skill_scores"].to_csv(boot_path, index=False)
    logger.info("Wrote MAPD fairness CSVs:\n  %s\n  %s", point_path, boot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
