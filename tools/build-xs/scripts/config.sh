#!/usr/bin/env bash
# Shared config for the OpenMHC-XS build. Source it: `source scripts/config.sh`
# NOTE: do not `set -e` here (this file is sourced); each executable script sets its own.

# --- paths (all overridable from the environment before sourcing) ---
export BUILD="${BUILD:-/scratch/users/eggert/openmhc-xs-build}"
export FULL="${FULL:-$BUILD/full}"                # rclone mirror of GDrive OpenMHC-Full (downloaded, untouched)
export EXTRACT="${EXTRACT:-$BUILD/full_extracted}"   # full tree with archives reconstructed/extracted in place
export STAGE="${STAGE:-$BUILD/xs_staged}"          # filtered full-layout XS tree (extracted)
export UPLOAD="${UPLOAD:-$BUILD/OpenMHC-XS}"        # final upload tree: archives/*.tar.gz + small data/ files
export SPLIT="$BUILD/sharable_users_seed42_2026_xs.json"        # 593-user XS split (the filter key)
export FULLSPLIT="$BUILD/sharable_users_seed42_2026_full.json"  # 11,894 users (json-shape detection only)
export REPORT="$BUILD/_xs_build_report.json"

# --- GDrive source ---
export GDRIVE_REMOTE="gdrive"
export GDRIVE_FOLDER_ID="1YH8BozsH5VW_9MUx8UEIWxPULJ1Wl-zB"   # OpenMHC-Full
export GDRIVE_FOLDER_NAME="OpenMHC-Full"

# --- Dataverse target (override per-run: DOI=doi:10.7910/DVN/ZYMJF6 bash 06_upload.sh) ---
export DOI="${DOI:-doi:10.7910/DVN/ZYMJF6}"
export DATAVERSE_BASE="https://dataverse.harvard.edu"
export TOKEN_FILE="$HOME/.dataverse_token"

# --- python modules for compute stages (NO datasets dep: HF dirs are filtered with pyarrow) ---
export PY_MODULES="python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312"

# Keep HuggingFace/datasets caches OFF $HOME (15 GB NFS). Callers may override.
export HF_HOME="${HF_HOME:-$BUILD/.hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BUILD/.hf/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BUILD/.cache}"

load_py() { module load $PY_MODULES; }
