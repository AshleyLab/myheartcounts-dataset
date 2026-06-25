#!/bin/bash
# LSM2 dense weekly imputation eval (Masked Autoencoder ViT, 10080-step weekly).
# Same wrapper class as the daily ``lsm2`` sbatch — weekly sizing
# (seq_length=10080, patch_size=60) is restored from the Lightning ckpt's
# hparams. Holds activations across the full 7-day window; request 32GB
# tier to match the sparse-weekly variant for headroom.
#
# Out of the main paper table (not in submit_all.sh / sweep_methods.yaml);
# run manually to produce pairs/ under ${RUNS_ROOT}/lsm2_weekly.

#SBATCH --job-name=mhc-impute-lsm2-weekly
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
# GPU_MEM tiers on Sherlock: 11,12,16,24,32,48,80 (40 is not a real label).
#SBATCH --constraint=GPU_MEM:32GB
#SBATCH --output=/scratch/users/%u/logs/openmhc/lsm2_weekly_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/lsm2_weekly_%j.err

# shellcheck disable=SC1091
# Use absolute path: sbatch copies the script to /var/spool/slurmd at run
# time, so $BASH_SOURCE doesn't point at the in-repo sibling _common.sh.
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

nvidia-smi || true

mhc-impute-eval \
  method=lsm2 \
  hydra.run.dir="${RUNS_ROOT}/lsm2_weekly" \
  output.results_dir="${RUNS_ROOT}/lsm2_weekly" \
  method.release_dir="${RELEASES}/lsm2_weekly" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.n_days=7 \
  data.batch_size=500 \
  data.num_workers=4 \
  data.num_eval_workers=1 \
  method.inference_batch_size=8 \
  +method.lsm2.inference_dropout_removal_ratio=0.0 \
  seed=42
