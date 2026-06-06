#!/usr/bin/env bash
# Shared setup for Simurgh (SC) forecasting-eval jobs.
#
# Mirrors jobs/sherlock/forecasting_eval/_common.sh but adapted for SC:
#   - conda env activation instead of module/venv,
#   - explicit MHC_DATA_DIR (no ~/.cache fallback on SC),
#   - BLAS threads pinned to 1 (statsforecast/ARIMA use joblib n_jobs=-1;
#     unpinned BLAS would oversubscribe cores),
#   - outputs under the repo (which already lives on /simurgh fast storage).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MHC_REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${REPO_DIR}"
mkdir -p logs/forecasting_eval

# --- Python env: conda (no environment modules on SC) ---
MHC_CONDA_BASE="${MHC_CONDA_BASE:-/simurgh/u/schuetzn/conda}"
MHC_CONDA_ENV="${MHC_CONDA_ENV:-openmhc}"
# shellcheck source=/dev/null
source "${MHC_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MHC_CONDA_ENV}"

export PYTHONNOUSERSITE=1
# Prepend src/ so the in-repo (edited) source is used regardless of how the
# package was installed. The mhc-forecast-eval console script resolves the
# entry point through this path.
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

# --- Pin BLAS threads (avoid joblib x BLAS oversubscription) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Dataset root: explicit on SC ---
if [[ -z "${MHC_DATA_DIR:-}" ]]; then
    export MHC_DATA_DIR="/simurgh/u/schuetzn/OpenMHC-Full/data"
fi

# --- Output roots ---
export RUNS_ROOT="${MHC_FORECAST_RUNS_ROOT:-${REPO_DIR}/results/forecasting_eval/simurgh}"
export RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_${SLURM_JOB_ID:-local}}"
mkdir -p "${RUNS_ROOT}"

echo "=== forecasting-eval common setup ==="
echo "REPO_DIR=${REPO_DIR}"
echo "conda env=${MHC_CONDA_ENV}  python=$(command -v python)"
echo "MHC_DATA_DIR=${MHC_DATA_DIR}"
echo "RUNS_ROOT=${RUNS_ROOT}"
echo "RUN_LABEL=${RUN_LABEL}"

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
