#!/bin/bash
# LSM2 daily imputation eval (Masked Autoencoder ViT, 1440-step daily).
# ~8-16 GB VRAM; request a 40 GB+ GPU (A100/L40S/H100) for headroom.

#SBATCH --job-name=mhc-impute-lsm2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
# GPU_MEM tiers on Sherlock: 11,12,16,24,32,48,80 (40 is not a real label).
# LSM2 daily needs ~8-16GB; 24GB tier is plenty with inference_batch_size=32.
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --output=/scratch/users/%u/logs/openmhc/lsm2_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/lsm2_%j.err

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nvidia-smi || true

mhc-impute-eval \
  method=lsm2 \
  hydra.run.dir="${RUNS_ROOT}/lsm2" \
  output.results_dir="${RUNS_ROOT}/lsm2" \
  method.release_dir="${RELEASES}/lsm2" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.batch_size=2500 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=32 \
  +method.lsm2.inference_dropout_removal_ratio=0.0 \
  seed=42
