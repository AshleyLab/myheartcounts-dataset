#!/usr/bin/env bash
#SBATCH --job-name=fc-autoarima
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_eval/autoarima_%j.out
#SBATCH --error=logs/forecasting_eval/autoarima_%j.err

# autoARIMA: the long pole. stepwise=False + n_jobs=-1 parallelises the per-series
# grid search across cores (diminishing past ~16). Resumable: re-submitting skips
# users whose parquet already exists (output.overwrite_existing_parquet=false).
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

echo "Start: $(date)  Host: $(hostname)"
echo "=== Running autoARIMA ==="
run_forecast_model autoARIMA "$@"
echo "End: $(date)"
