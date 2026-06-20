#!/usr/bin/env bash
#SBATCH --job-name=fc-paper-bootstrap
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --exclude=simurgh2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/forecasting_eval/paper_bootstrap_%j.out
#SBATCH --error=logs/forecasting_eval/paper_bootstrap_%j.err

# Full Layer-2 paper leaderboard from an EXISTING metrics tree: point skill + rank,
# paired user-bootstrap CIs (n_boot from the sweep), and the disparity-ratio
# fairness skill score (point + CIs) — all under the category-balanced channel→scope
# scheme. Runs run_paper_pipeline.py --skip-eval over the sweep's RUN_LABEL metrics
# and writes to a SEPARATE output dir so the canonical summary is never clobbered.
#
# Env knobs:
#   MHC_FORECAST_SWEEP     sweep config (default configs/paper/sweep_forecasting.yaml)
#   MHC_FORECAST_OUT_LABEL output summary subdir name (REQUIRED — keeps canonical safe)
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

SWEEP="${MHC_FORECAST_SWEEP:-configs/paper/sweep_forecasting.yaml}"
OUT_LABEL="${MHC_FORECAST_OUT_LABEL:?set MHC_FORECAST_OUT_LABEL (output summary subdir; do not reuse the canonical label)}"
OUT_DIR="${RUNS_ROOT}/summary/${OUT_LABEL}"
mkdir -p "${OUT_DIR}"

echo "=== run_paper_pipeline (skip-eval): skill + rank + bootstrap + fairness ==="
echo "sweep=${SWEEP}  ->  ${OUT_DIR}"
python scripts/paper_results/forecasting/run_paper_pipeline.py \
    --sweep-config "${SWEEP}" \
    --skip-eval \
    --output-root "${OUT_DIR}"

echo "=== Outputs under ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
