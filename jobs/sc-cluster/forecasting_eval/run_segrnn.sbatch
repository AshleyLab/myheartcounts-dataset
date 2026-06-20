#!/usr/bin/env bash
#SBATCH --job-name=fc-segrnn
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/forecasting_eval/segrnn_%j.out
#SBATCH --error=logs/forecasting_eval/segrnn_%j.err

# SegRNN (fine-tuned, from-scratch neural). Runs in the openmhc env (pypots).
# Requires MHC_FORECAST_SEGRNN_RELEASE_DIR pointing at the downloaded bundle.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)"
echo "=== Running segrnn (fine-tuned) ==="
override="$(require_release_override segrnn)"
run_forecast_variant segrnn segrnn "${override}" "$@"
echo "End: $(date)"
