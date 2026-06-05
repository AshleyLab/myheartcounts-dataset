#!/usr/bin/env bash
# Shared setup for Sherlock forecasting-eval jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MHC_REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${REPO_DIR}"
mkdir -p logs/forecasting_eval

if command -v module >/dev/null 2>&1; then
    module purge
    module load "${MHC_PYTHON_MODULE:-python/3.12.1}" || true
    if [[ "${MHC_LOAD_CUDA:-0}" == "1" ]]; then
        module load "${MHC_CUDA_MODULE:-cuda/12.2.0}" || true
    fi
fi

if [[ -n "${MHC_VENV:-}" && -f "${MHC_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${MHC_VENV}/bin/activate"
elif [[ -f "/scratch/users/${USER}/envs/mhc-benchmark/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "/scratch/users/${USER}/envs/mhc-benchmark/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
if [[ -z "${MHC_DATA_DIR:-}" ]]; then
    if [[ -d "${HOME}/.cache/openmhc/data-full" ]]; then
        export MHC_DATA_DIR="${HOME}/.cache/openmhc/data-full"
    else
        export MHC_DATA_DIR="${HOME}/.cache/openmhc/data"
    fi
fi
export RUNS_ROOT="${MHC_FORECAST_RUNS_ROOT:-${REPO_DIR}/results/forecasting_eval/sherlock}"
export RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_${SLURM_JOB_ID:-local}}"

mkdir -p "${RUNS_ROOT}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_KEY="$(awk '/machine api\.wandb\.ai/{found=1;next} found && /password/{print $2;exit}' "${HOME}/.netrc" 2>/dev/null || true)"
    export WANDB_API_KEY="${WANDB_KEY:-}"
fi
export WANDB_DIR="${WANDB_DIR:-/scratch/users/${USER}/wandb_data/run_data}"

run_forecast_model() {
    local model="$1"
    shift

    local model_root="${RUNS_ROOT}/${model}"
    mkdir -p "${model_root}"

    mhc-forecast-eval \
        "model=${model}" \
        "experiment_name=${RUN_LABEL}" \
        "output.results_dir=${model_root}/predictions" \
        "output.overwrite_existing_parquet=${MHC_FORECAST_OVERWRITE:-false}" \
        "hydra.run.dir=${model_root}/hydra" \
        "hydra.job.chdir=false" \
        "$@"
}

release_override() {
    local model="$1"
    local env_name="MHC_FORECAST_${model^^}_RELEASE_DIR"
    local release_dir="${!env_name:-}"
    if [[ -n "${release_dir}" ]]; then
        printf '%s\n' "model.release_dir=${release_dir}"
    fi
}

require_release_override() {
    local model="$1"
    local override
    override="$(release_override "${model}")"
    if [[ -z "${override}" ]]; then
        echo "ERROR: set MHC_FORECAST_${model^^}_RELEASE_DIR for ${model}." >&2
        exit 2
    fi
    printf '%s\n' "${override}"
}
