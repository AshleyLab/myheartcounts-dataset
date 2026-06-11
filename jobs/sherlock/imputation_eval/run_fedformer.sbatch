#!/bin/bash
# FEDformer imputation eval (PyPOTS Transformer, ~4-8 GB VRAM).

#SBATCH --job-name=mhc-impute-fedformer
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --output=/scratch/users/%u/logs/openmhc/fedformer_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/fedformer_%j.err

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

nvidia-smi || true

mhc-impute-eval \
  method=fedformer \
  hydra.run.dir="${RUNS_ROOT}/fedformer" \
  output.results_dir="${RUNS_ROOT}/fedformer" \
  method.release_dir="${RELEASES}/fedformer" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.batch_size=2000 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=64 \
  evaluation.save_pairs=true \
  bootstrap=on \
  seed=42
