#!/usr/bin/env bash
#SBATCH --job-name=fc-dlinear
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/forecasting_eval/dlinear_%j.out
#SBATCH --error=logs/forecasting_eval/dlinear_%j.err

# DLinear (fine-tuned, from-scratch neural). Runs in the openmhc env (pypots).
# Requires MHC_FORECAST_DLINEAR_RELEASE_DIR pointing at the downloaded bundle.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)"
echo "=== Running dlinear (fine-tuned) ==="
override="$(require_release_override dlinear)"
run_forecast_variant dlinear dlinear "${override}" "$@"
echo "End: $(date)"
