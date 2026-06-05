#!/usr/bin/env bash
#SBATCH --job-name=forecast-baselines
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_eval/baselines_%j.out
#SBATCH --error=logs/forecasting_eval/baselines_%j.err

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== Forecasting baselines ==="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Runs root: ${RUNS_ROOT}"

for model in seasonal_naive seasonal_naive_average_history autoARIMA autoETS; do
    echo ""
    echo "=== Running ${model} ==="
    run_forecast_model "${model}" "$@"
done

echo "End: $(date)"
