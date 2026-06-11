# shellcheck shell=bash
# Common environment for every imputation-train sbatch on Sherlock.
# Source from each sbatch as:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#
# Idempotent. Safe to source multiple times. All overridable knobs come
# from ``../jobs/sherlock/_env.sh`` (REPO, SCRATCH_RUN_ROOT, LOGS,
# MHC_CACHE, MHC_DATA_DIR, WANDB_DIR, RELEASES).

set -euo pipefail

# --- Shared defaults ----------------------------------------------------------
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_COMMON_DIR}/../_env.sh"

# Same dataset cache the eval pipeline uses — train and eval share splits exactly.
export DAILY_HF_DIR=${MHC_CACHE}/processed/daily_hf
export SPLIT_FILE=${MHC_CACHE}/splits/sharable_users_seed42_2026.json
# normalization_stats.json is read from <MHC_CACHE>/processed/ automatically.

# Output roots. Everything under TRAIN_BASE; nothing in $HOME.
: "${TRAIN_BASE:=${SCRATCH_RUN_ROOT}/openmhc-imputation-train}"
export TRAIN_BASE
export H5_CACHE=${TRAIN_BASE}/h5         # H5 export cache (content-addressed subdirs)
export RUNS_ROOT=${TRAIN_BASE}/runs      # one subdir per training run (PyPOTS' saving_path)
export RELEASES_ROOT=${TRAIN_BASE}/releases   # one openmhc release bundle per trained model

# --- Hydra / W&B housekeeping -------------------------------------------------
export HYDRA_FULL_ERROR=1

# Pull WANDB_API_KEY from ~/.netrc so wandb.init() works in batch jobs without
# a TTY login. (Matches MHC-benchmark/jobs/.../train_pypots.sbatch.)
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_KEY=$(awk '/machine api\.wandb\.ai/{found=1;next} found && /password/{print $2;exit}' \
        "$HOME/.netrc" 2>/dev/null || true)
    if [ -n "${WANDB_KEY:-}" ]; then
        export WANDB_API_KEY="$WANDB_KEY"
    fi
fi

mkdir -p "$TRAIN_BASE" "$H5_CACHE" "$RUNS_ROOT" "$RELEASES_ROOT" "$LOGS" "$WANDB_DIR"

# --- venv activation ----------------------------------------------------------
# In-repo Sherlock-aware activator (LD_LIBRARY_PATH, PYTHONPATH cleanup, etc.).
cd "$REPO"
# shellcheck disable=SC1091
source scripts/dev/activate-openmhc.sh

module load cuda/12.2.0 2>/dev/null || true
