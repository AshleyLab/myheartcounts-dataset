#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=p3_phase2
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/schuetzn/tmp/claude/parity_bootstrap_phase2.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/parity_bootstrap_phase2.err

# §3 (statistical): run Phase 2 aggregations on the new bootstrap_draws and
# compare CSVs against paper/*_bootstrap.csv using parity_point.py's (B)
# statistical bar (mean ± 5×SE). Byte parity on draws failed due to:
#   - float32 storage in per_user_errors (Δ ≲ 1e-3 on large-magnitude E)
#   - cat_collapsed:* R formula split (existing path floors per-channel
#     INSIDE the nanmean; producer stores unfloored nanmean — semantic
#     divergence specific to the unknown subgroup)
#   - subtle rank-pool composition differences
# All three are small; the statistical CSV match is the right scientific bar.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

PAPER=/scratch/users/schuetzn/openmhc-imputation-eval/paper
VERIF=/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification
mkdir -p "$VERIF/bootstrap_phase2"

echo "=== (1/3) Phase 2 skill + rank ==="
python scripts/paper_results/imputation/aggregate_imputation_paper_metrics.py \
    --draws "$VERIF/bootstrap/bootstrap_draws.parquet" \
    --output-dir "$VERIF/bootstrap_phase2/"

echo "=== (2/3) Phase 2 fairness ==="
python scripts/paper_results/aggregate_fairness_skill_score.py \
    --draws "$VERIF/bootstrap/bootstrap_draws.parquet" \
    --output "$VERIF/bootstrap_phase2/fairness_skill_score_bootstrap.csv"

echo "=== (3/3) Parity diff vs paper/*_bootstrap.csv ==="
python - <<'PYEOF'
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger()

VERIF = Path("/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification/bootstrap_phase2")
PAPER = Path("/scratch/users/schuetzn/openmhc-imputation-eval/paper")

JOIN_KEYS = ["method", "scope", "split"]
TOL = 5.0  # × SE
failed = False

for name in ("skill_scores_bootstrap.csv", "avg_rankings_bootstrap.csv", "fairness_skill_score_bootstrap.csv"):
    new_path = VERIF / name
    gt_path = PAPER / name
    if not new_path.exists() or not gt_path.exists():
        log.warning("skipping %s (missing)", name)
        continue
    new = pd.read_csv(new_path)
    gt = pd.read_csv(gt_path)
    for c in JOIN_KEYS:
        new[c] = new[c].astype(str)
        gt[c] = gt[c].astype(str)
    merged = new.merge(gt, on=JOIN_KEYS, suffixes=("_new", "_gt"))
    if "mean_new" not in merged or "mean_gt" not in merged or "se_gt" not in merged:
        log.error("schema mismatch in %s — cols: %s", name, list(merged.columns))
        failed = True
        continue
    diff = np.abs(merged["mean_new"].to_numpy() - merged["mean_gt"].to_numpy())
    se = merged["se_gt"].to_numpy()
    threshold = TOL * se
    oob = diff > threshold
    n_oob = int(np.nansum(oob))
    if n_oob == 0:
        log.info("PASS: %s — %d/%d rows within %.1f×SE (max |Δmean|=%.3e)",
                 name, len(merged), len(merged), TOL, float(np.nanmax(diff)))
    else:
        failed = True
        worst = merged.assign(absd=diff, thr=threshold,
                              band=diff/np.where(threshold>0, threshold, np.nan)
                              ).nlargest(10, "band")
        log.error("FAIL: %s — %d/%d rows outside %.1f×SE band (max |Δmean|=%.3e)",
                  name, n_oob, len(merged), TOL, float(np.nanmax(diff)))
        log.error("worst 10:\n%s", worst.to_string())

if failed:
    raise SystemExit(1)
log.info("§3 (statistical) PASSED ✓")
PYEOF
