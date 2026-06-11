# shellcheck shell=bash
# Common environment for every imputation-eval sbatch on Sherlock.
# Source from each sbatch as:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#
# Idempotent. Safe to source multiple times. All overridable knobs come
# from ``../jobs/sherlock/_env.sh`` (REPO, SCRATCH_RUN_ROOT, LOGS,
# MHC_CACHE, MHC_DATA_DIR, WANDB_DIR, RELEASES).

set -euo pipefail

# --- Shared defaults ----------------------------------------------------------
# Resolve location of this file, source the shared env defaults.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_COMMON_DIR}/../_env.sh"

# --- Eval-specific paths ------------------------------------------------------
# Dataset cache populated by ``openmhc.download_dataset(version="full")``.
# 292 GB, 11894 users, 2.82M daily rows. version="full".
export DAILY_HF_DIR=${MHC_CACHE}/processed/daily_hf
export SPLIT_FILE=${MHC_CACHE}/splits/sharable_users_seed42_2026.json

# Output roots. Everything under OUT_BASE; nothing in $HOME.
: "${OUT_BASE:=${SCRATCH_RUN_ROOT}/openmhc-imputation-eval}"
export OUT_BASE
export RUNS_ROOT=${OUT_BASE}/runs       # one subdir per method (sweep config layout)
export PAPER_OUT=${OUT_BASE}/paper

# Mask files for strict parity with MHC-benchmark's max91d ablation.
# Materialized once from your DVC cache, if you have one.
export MASKS=${OUT_BASE}/masks/sharable_users_seed42_2026_max91d

# --- Hydra / W&B housekeeping -------------------------------------------------
export HYDRA_FULL_ERROR=1

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
