#!/usr/bin/env bash
# Shared setup for Sherlock forecasting-TRAIN jobs.
#
# Mirrors jobs/sherlock/forecasting_eval/_common.sh (module + venv activation,
# MHC_DATA_DIR resolution, W&B key from ~/.netrc) but for training: it adds
# output roots for PyPOTS run dirs + openmhc release bundles under $SCRATCH, and
# a run_forecast_train helper. The bundles are consumed by the eval jobs via
# MHC_FORECAST_<MODEL>_RELEASE_DIR=<release_dir>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MHC_REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${REPO_DIR}"
mkdir -p logs/forecasting_train

if command -v module >/dev/null 2>&1; then
    module purge
    module load "${MHC_PYTHON_MODULE:-python/3.12.1}" || true
    if [[ "${MHC_LOAD_CUDA:-1}" == "1" ]]; then
        module load "${MHC_CUDA_MODULE:-cuda/12.2.0}" || true
    fi
fi

# In-repo Sherlock-aware activator (LD_LIBRARY_PATH, PYTHONPATH cleanup, venv).
# shellcheck source=/dev/null
source scripts/dev/activate-openmhc.sh

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# --- Dataset root (train + eval share the same cache) ---
if [[ -z "${MHC_DATA_DIR:-}" ]]; then
    if [[ -d "${HOME}/.cache/openmhc/data-full" ]]; then
        export MHC_DATA_DIR="${HOME}/.cache/openmhc/data-full"
    else
        export MHC_DATA_DIR="${HOME}/.cache/openmhc/data"
    fi
fi

# --- Output roots (keep off the home quota) ---
export TRAIN_BASE="${MHC_FORECAST_TRAIN_BASE:-/scratch/users/${USER}/openmhc-forecasting-train}"
export RUNS_ROOT="${TRAIN_BASE}/runs"
export RELEASES_ROOT="${TRAIN_BASE}/releases"
mkdir -p "${RUNS_ROOT}" "${RELEASES_ROOT}"

# --- W&B (optional). Pull key from ~/.netrc for headless jobs. ---
export WANDB_DIR="${WANDB_DIR:-/scratch/users/${USER}/wandb_data/openmhc}"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_KEY="$(awk '/machine api\.wandb\.ai/{found=1;next} found && /password/{print $2;exit}' \
        "${HOME}/.netrc" 2>/dev/null || true)"
    [[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="${WANDB_KEY}"
fi

echo "=== forecasting-train common setup ==="
echo "REPO_DIR=${REPO_DIR}"
echo "python=$(command -v python)"
echo "MHC_DATA_DIR=${MHC_DATA_DIR}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "RELEASES_ROOT=${RELEASES_ROOT}"

# Train one model and emit a timestamped release bundle. Prints the release dir
# on the final line as ``RELEASE_DIR=<path>``.
run_forecast_train() {
    local model="$1"
    shift
    local ts saving_path release_dir
    ts="$(date +%Y%m%d_%H%M%S)"
    saving_path="${RUNS_ROOT}/${model}_${ts}"
    release_dir="${RELEASES_ROOT}/${model}_${ts}"

    mhc-forecast-train \
        "model=${model}" \
        "seed=${MHC_FORECAST_SEED:-42}" \
        "output.saving_path=${saving_path}" \
        "output.release_dir=${release_dir}" \
        "output.wandb_enabled=${MHC_FORECAST_WANDB:-true}" \
        "$@"

    echo "RELEASE_DIR=${release_dir}"
}
