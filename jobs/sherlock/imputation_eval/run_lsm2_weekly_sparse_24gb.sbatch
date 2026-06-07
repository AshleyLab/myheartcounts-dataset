#!/bin/bash
# LSM2 weekly sparse imputation eval — 24GB GPU fallback variant.
#
# Why this exists: the GPU_MEM:32GB constraint on the canonical sbatch
# (run_lsm2_weekly_sparse.sbatch) restricts placement to ~5 Sherlock nodes,
# all heavily contended; jobid 27542937 has been pending ~46h with
# StartTime=Unknown. This variant relaxes the GPU constraint to any GPU
# (typically lands on 24GB tier) and mirrors the pairs-only + lowered-buffer
# config that made dlinear_weekly fit in 128G mem.
#
# Trade-offs vs the canonical sbatch:
#   - GPU memory floor 24GB instead of 32GB. The model itself is ~12-20 GB
#     activation footprint; should fit on 24GB at inference_batch_size <= 8.
#   - Pairs-only Phase 1 (no in-eval metrics/bootstrap accumulator). Metrics
#     and per-imputer CIs are recomputed by run_paper_pipeline.py Phase 2
#     from the saved pairs — no information lost, identical to MHC-benchmark
#     and dlinear_weekly setup.
#   - Lowered data.batch_size and num_workers to stay well under 128G mem.
#
# Run alongside the 32GB-constrained job — whichever finishes first wins;
# the other should be cancelled after a successful paper_bootstrap.

#SBATCH --job-name=mhc-impute-lsm2-weekly-sparse-24gb
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
# NOTE: no GPU_MEM constraint — let SLURM place us on whichever GPU is free.
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/lsm2_weekly_sparse_24gb_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/lsm2_weekly_sparse_24gb_%j.err

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
  data.batch_size=500 \
  data.num_workers=4 \
  data.num_eval_workers=1 \
  method.inference_batch_size=8 \
  evaluation.save_pairs=true \
  evaluation.compute_metrics=false \
  bootstrap=off \
  seed=42
