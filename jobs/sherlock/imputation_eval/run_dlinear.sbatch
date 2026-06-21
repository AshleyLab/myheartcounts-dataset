#!/bin/bash
# DLinear imputation eval (PyPOTS linear, ~1-2 GB VRAM — smallest neural model).
# Shorter time + smaller mem than the others.

#SBATCH --job-name=mhc-impute-dlinear
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=/scratch/users/%u/logs/openmhc/dlinear_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/dlinear_%j.err

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nvidia-smi || true

mhc-impute-eval \
  method=dlinear \
  hydra.run.dir="${RUNS_ROOT}/dlinear" \
  output.results_dir="${RUNS_ROOT}/dlinear" \
  method.release_dir="${RELEASES}/dlinear" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.batch_size=2000 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=128 \
  seed=42
