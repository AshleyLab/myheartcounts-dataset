#!/bin/bash
# LSM2 dense weekly imputation eval — 48GB GPU tier variant.
#
# Why this exists: the canonical sbatch (run_lsm2_weekly.sbatch) requests
# GPU_MEM:32GB. The dense 7-day variant has full quadratic attention across
# all weekly tokens (sparse variant deliberately avoids this), so the 32GB
# tier may be tight at inference_batch_size=16. This variant requests the
# 48GB tier for headroom. Run alongside the 32GB-constrained job — whichever
# finishes first wins; cancel the other once one starts producing pairs.
#
# Output dir is identical to the canonical sbatch so pair files land at the
# same path regardless of which tier won (mirrors the lsm2_weekly_sparse_24gb
# precedent).

#SBATCH --job-name=mhc-impute-lsm2-weekly-48gb
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
# GPU_MEM tiers on Sherlock: 11,12,16,24,32,48,80 (40 is not a real label).
#SBATCH --constraint=GPU_MEM:48GB
#SBATCH --output=/scratch/users/%u/logs/openmhc/lsm2_weekly_48gb_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/lsm2_weekly_48gb_%j.err

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
  data.batch_size=2500 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=16 \
  +method.lsm2.inference_dropout_removal_ratio=0.0 \
  seed=42
