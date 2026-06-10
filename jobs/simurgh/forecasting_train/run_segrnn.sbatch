#!/usr/bin/env bash
#SBATCH --job-name=fc-train-segrnn
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_train/segrnn_%j.out
#SBATCH --error=logs/forecasting_train/segrnn_%j.err

# Train SegRNN (PyPOTS forecasting) on padded short+full histories and emit a
# release bundle consumable by mhc-forecast-eval via model.release_dir=.
#
# Training recipe pinned to the published checkpoint
# (releases-fc/openmhc-segrnn-fc/training_config.json): epochs=30,
# batch_size=256, optimizer_lr=0.002025. Architecture lives in the model YAML.
# Only change vs the original is include_short_history=true (the config default).
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/simurgh/forecasting_train/_common.sh"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)"
echo "=== Training segrnn ==="
run_forecast_train segrnn \
    training.epochs=30 \
    training.batch_size=256 \
    training.optimizer_lr=0.002025 \
    training.num_workers=8 \
    "$@"
echo "End: $(date)"
