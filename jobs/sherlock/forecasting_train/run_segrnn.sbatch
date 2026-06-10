#!/bin/bash
#SBATCH --job-name=fc-train-segrnn
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=GPU_BRD:TESLA
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/train_fc_segrnn_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/train_fc_segrnn_%j.err

# Train SegRNN (PyPOTS forecasting); emit a release bundle for mhc-forecast-eval.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sherlock/forecasting_train/_common.sh"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)"
echo "=== Training segrnn ==="
run_forecast_train segrnn "$@"
echo "End: $(date)"
