#!/usr/bin/env bash
#SBATCH --job-name=fc-naive
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/forecasting_eval/naive_%j.out
#SBATCH --error=logs/forecasting_eval/naive_%j.err

# Naive baselines: pure-numpy, single-threaded — gain nothing from extra cores.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

echo "Start: $(date)  Host: $(hostname)"
for model in seasonal_naive; do
    echo ""
    echo "=== Running ${model} ==="
    run_forecast_model "${model}" "$@"
done
echo "End: $(date)"
