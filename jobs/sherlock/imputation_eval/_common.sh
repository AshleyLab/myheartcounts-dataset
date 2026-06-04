# shellcheck shell=bash
# Common environment for every imputation-eval sbatch on Sherlock.
# Source from each sbatch as:
#   source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh
#
# Idempotent. Safe to source multiple times.

set -euo pipefail

# --- Paths --------------------------------------------------------------------
export REPO=/home/users/schuetzn/myheartcounts-dataset

# Dataset cache populated by `openmhc.download_dataset(version="full")`.
# 292 GB, 11894 users, 2.82M daily rows. version="full".
export MHC_CACHE=/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-full
export MHC_DATA_DIR=${MHC_CACHE}          # required by reference-imputer fit-phase loaders
export DAILY_HF_DIR=${MHC_CACHE}/processed/daily_hf
export SPLIT_FILE=${MHC_CACHE}/splits/sharable_users_seed42_2026.json

# Output roots. Everything under OUT_BASE; nothing in $HOME.
export OUT_BASE=/scratch/users/schuetzn/openmhc-imputation-eval
export RUNS_ROOT=${OUT_BASE}/runs       # one subdir per method (sweep config layout)
export PAPER_OUT=${OUT_BASE}/paper
export LOGS=/scratch/users/schuetzn/logs/openmhc

# W&B-downloaded paper checkpoint bundles (populated by 00_setup.sh).
export RELEASES=/scratch/users/schuetzn/releases

# Mask files for strict parity with MHC-benchmark's max91d ablation.
# Copied from the DVC cache (the in-repo /data/imputation/masks/ paths are
# LFS pointer stubs on this checkout, since git-lfs isn't installed on
# Sherlock). Materialized once by hand from /home/users/schuetzn/MHC-benchmark/
# (DVC symlinks -> /scratch/users/schuetzn/dvc-cache/).
export MASKS=${OUT_BASE}/masks/sharable_users_seed42_2026_max91d

# --- Hydra / W&B housekeeping -------------------------------------------------
export HYDRA_FULL_ERROR=1
export WANDB_DIR=/scratch/users/schuetzn/wandb_data/openmhc

mkdir -p "$OUT_BASE" "$RUNS_ROOT" "$PAPER_OUT" "$LOGS" "$WANDB_DIR"

# --- venv activation ----------------------------------------------------------
# Use the in-repo Sherlock-aware activator. It:
#   - prepends /share/software python+openssl libs to LD_LIBRARY_PATH
#   - unsets PYTHONPATH (else py3.9 site-packages shadow the venv)
#   - unsets CC=mpicc / CXX=mpic++ / F77 / FC (else source builds fail)
cd "$REPO"
# shellcheck disable=SC1091
source scripts/dev/activate-openmhc.sh

# CUDA module — harmless on CPU partitions, needed on gpu.
module load cuda/12.2.0 2>/dev/null || true
