#!/usr/bin/env bash
# Shared setup for Simurgh (SC) forecasting-TRAIN jobs.
#
# Mirrors jobs/sc-cluster/forecasting_eval/_common.sh (conda env, explicit
# MHC_DATA_DIR, HF cache off the home quota) but for training: it adds output
# roots for PyPOTS run dirs + openmhc release bundles, and a run_forecast_train
# helper. The bundles it produces are consumed by the eval jobs via
# MHC_FORECAST_<MODEL>_RELEASE_DIR=<release_dir>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MHC_REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${REPO_DIR}"
mkdir -p logs/forecasting_train

# --- Python env: conda (no environment modules on SC) ---
MHC_CONDA_BASE="${MHC_CONDA_BASE:-/simurgh/u/schuetzn/conda}"
MHC_CONDA_ENV="${MHC_CONDA_ENV:-openmhc}"
# shellcheck source=/dev/null
source "${MHC_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MHC_CONDA_ENV}"

export PYTHONNOUSERSITE=1
# Prepend src/ so the in-repo (edited) source is used regardless of how the
# package was installed. The mhc-forecast-train console script resolves the
# entry point through this path.
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# --- Dataset root: explicit on SC (train + eval share the same cache) ---
if [[ -z "${MHC_DATA_DIR:-}" ]]; then
    export MHC_DATA_DIR="/simurgh/u/schuetzn/OpenMHC-Full/data"
fi

# --- HF cache: keep downloads off the home quota ---
export HF_HOME="${HF_HOME:-/simurgh/u/schuetzn/.cache/huggingface}"

# --- Output roots ---
export TRAIN_BASE="${MHC_FORECAST_TRAIN_BASE:-${REPO_DIR}/results/forecasting_train/simurgh}"
export RUNS_ROOT="${TRAIN_BASE}/runs"          # PyPOTS saving_path per run
export RELEASES_ROOT="${TRAIN_BASE}/releases"  # one openmhc release bundle per trained model
mkdir -p "${RUNS_ROOT}" "${RELEASES_ROOT}"

# --- W&B (optional). Pull key from ~/.netrc for headless jobs. ---
export WANDB_DIR="${WANDB_DIR:-${TRAIN_BASE}/wandb}"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_KEY="$(awk '/machine api\.wandb\.ai/{found=1;next} found && /password/{print $2;exit}' \
        "${HOME}/.netrc" 2>/dev/null || true)"
    [[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="${WANDB_KEY}"
fi

echo "=== forecasting-train common setup ==="
echo "REPO_DIR=${REPO_DIR}"
echo "conda env=${MHC_CONDA_ENV}  python=$(command -v python)"
echo "MHC_DATA_DIR=${MHC_DATA_DIR}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "RELEASES_ROOT=${RELEASES_ROOT}"

# Train one model and emit a timestamped release bundle. Prints the release dir
# on the final line as ``RELEASE_DIR=<path>`` so it's easy to grep from logs and
# feed into the eval job's MHC_FORECAST_<MODEL>_RELEASE_DIR.
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
