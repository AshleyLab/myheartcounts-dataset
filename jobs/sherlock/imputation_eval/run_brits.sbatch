#!/bin/bash
# BRITS imputation eval (PyPOTS RNN, ~2-4 GB VRAM).
# Resources mirror MHC-benchmark/.../run_imputation_brits.sbatch.

#SBATCH --job-name=mhc-impute-brits
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/brits_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/brits_%j.err

# shellcheck disable=SC1091
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

nvidia-smi || true

mhc-impute-eval \
  method=brits \
  hydra.run.dir="${RUNS_ROOT}/brits" \
  output.results_dir="${RUNS_ROOT}/brits" \
  method.release_dir="${RELEASES}/brits" \
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
