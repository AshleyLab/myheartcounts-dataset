#!/bin/bash
# LSM2 weekly sparse imputation eval (per-day encoder + sparse cross-day
# decoder). ~12-20 GB VRAM; request a 40 GB+ GPU.

#SBATCH --job-name=mhc-impute-lsm2-weekly-sparse
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
# GPU_MEM tiers on Sherlock: 11,12,16,24,32,48,80 (40 is not a real label).
# Weekly-sparse keeps activations across 7 days; request 32GB tier for headroom.
#SBATCH --constraint=GPU_MEM:32GB
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/lsm2_weekly_sparse_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/lsm2_weekly_sparse_%j.err

# shellcheck disable=SC1091
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

nvidia-smi || true

mhc-impute-eval \
  method=lsm2_weekly_sparse \
  hydra.run.dir="${RUNS_ROOT}/lsm2_weekly_sparse" \
  output.results_dir="${RUNS_ROOT}/lsm2_weekly_sparse" \
  method.release_dir="${RELEASES}/lsm2_weekly_sparse" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.n_days=7 \
  data.batch_size=2500 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=16 \
  evaluation.save_pairs=true \
  bootstrap=on \
  seed=42
