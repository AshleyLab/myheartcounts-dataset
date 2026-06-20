# shellcheck shell=bash
#
# Defaults for the OpenMHC Sherlock job-script environment.
# Sourced from each ``_common.sh``. All vars use ``${VAR:=default}`` so
# the caller can override any of them by exporting before ``sbatch``
# (or by writing them into ``~/.mhc-sherlock.env`` and sourcing that
# from their shell rc).
#
# Conventions:
# - ``REPO`` defaults to the path resolved from this script's location
#   (assumes you are running from a clean clone of ``myheartcounts-dataset``).
# - All scratch-based paths default to ``/scratch/users/$USER/...`` so
#   anyone with a Sherlock account gets working defaults without editing.
# - Idempotent. Safe to source multiple times.

: "${USER:=$(id -un)}"

# Repo root. If REPO is unset, resolve it from this script's location:
# jobs/sherlock/_env.sh -> repo root (two levels up).
if [[ -z "${REPO:-}" ]]; then
    _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "${_ENV_DIR}/../.." && pwd)"
fi
export REPO

# Per-user scratch root and derived paths. Override SCRATCH_RUN_ROOT to
# point everything below at a different location in one shot.
: "${SCRATCH_RUN_ROOT:=/scratch/users/${USER}}"
: "${LOGS:=${SCRATCH_RUN_ROOT}/logs/openmhc}"
: "${MHC_CACHE:=${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-full}"
: "${MHC_DATA_DIR:=${MHC_CACHE}}"
: "${WANDB_DIR:=${SCRATCH_RUN_ROOT}/wandb_data/openmhc}"
: "${RELEASES:=${SCRATCH_RUN_ROOT}/releases}"
export SCRATCH_RUN_ROOT LOGS MHC_CACHE MHC_DATA_DIR WANDB_DIR RELEASES
