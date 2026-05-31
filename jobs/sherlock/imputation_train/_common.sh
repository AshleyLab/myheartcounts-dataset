# shellcheck shell=bash
# Common environment for every imputation-train sbatch on Sherlock.
# Source from each sbatch as:
#   source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_train/_common.sh
#
# Idempotent. Safe to source multiple times.

set -euo pipefail

# --- Paths --------------------------------------------------------------------
export REPO=/home/users/schuetzn/myheartcounts-dataset

# Same dataset cache the eval pipeline uses — train and eval share splits exactly.
export MHC_CACHE=/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-full
export MHC_DATA_DIR=${MHC_CACHE}
export DAILY_HF_DIR=${MHC_CACHE}/processed/daily_hf
export SPLIT_FILE=${MHC_CACHE}/splits/sharable_users_seed42_2026.json
# normalization_stats.json is read from <MHC_CACHE>/processed/ automatically.

# Output roots. Everything under TRAIN_BASE; nothing in $HOME.
export TRAIN_BASE=/scratch/users/schuetzn/openmhc-imputation-train
export H5_CACHE=${TRAIN_BASE}/h5         # H5 export cache (content-addressed subdirs)
export RUNS_ROOT=${TRAIN_BASE}/runs      # one subdir per training run (PyPOTS' saving_path)
export RELEASES_ROOT=${TRAIN_BASE}/releases   # one openmhc release bundle per trained model
export LOGS=/scratch/users/schuetzn/logs/openmhc

# --- Hydra / W&B housekeeping -------------------------------------------------
export HYDRA_FULL_ERROR=1
export WANDB_DIR=/scratch/users/schuetzn/wandb_data/openmhc

mkdir -p "$TRAIN_BASE" "$H5_CACHE" "$RUNS_ROOT" "$RELEASES_ROOT" "$LOGS" "$WANDB_DIR"

# --- venv activation ----------------------------------------------------------
# In-repo Sherlock-aware activator (LD_LIBRARY_PATH, PYTHONPATH cleanup, etc.).
cd "$REPO"
# shellcheck disable=SC1091
source scripts/dev/activate-openmhc.sh

module load cuda/12.2.0 2>/dev/null || true
