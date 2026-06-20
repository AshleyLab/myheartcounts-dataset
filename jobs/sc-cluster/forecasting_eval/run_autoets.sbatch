#!/usr/bin/env bash
#SBATCH --job-name=fc-autoets
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_eval/autoets_%j.out
#SBATCH --error=logs/forecasting_eval/autoets_%j.err

# autoETS: n_jobs=-1 over a small ETS grid — modest core benefit, saturates ~8.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

echo "Start: $(date)  Host: $(hostname)"
echo "=== Running autoETS ==="
run_forecast_model autoETS "$@"
echo "End: $(date)"
