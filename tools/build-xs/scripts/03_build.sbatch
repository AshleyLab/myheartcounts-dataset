#!/usr/bin/env bash
#SBATCH --job-name=openmhc-xs-build
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/users/eggert/openmhc-xs-build/build.%j.out
#SBATCH --error=/scratch/users/eggert/openmhc-xs-build/build.%j.err
# Heavy stage: extract -> filter -> repackage -> verify. Submit with: sbatch 03_build.sbatch
# Requires $FULL already downloaded (run 01_download.sh on the DTN first).
set -euo pipefail
# NOTE: under sbatch, $0 is a spool copy, so dirname "$0" is wrong. Resolve via submit dir
# (we submit from scripts/) with a hardcoded fallback.
HERE="${SLURM_SUBMIT_DIR:-}"
[ -f "$HERE/config.sh" ] || HERE="/scratch/users/eggert/openmhc-xs-build/scripts"
source "$HERE/config.sh"
load_py

# Use node-local SSD for the large extracted tree if it fits; else fall back to SCRATCH.
# daily_hf can expand a lot; default to SCRATCH for safety. Override by exporting EXTRACT.
export EXTRACT="${EXTRACT:-$BUILD/full_extracted}"
export HF_HOME="$BUILD/.hf"; mkdir -p "$HF_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # keep pyarrow/numpy within the allocation

echo "==== [1/4] extract ===="
rm -rf "$EXTRACT"; mkdir -p "$EXTRACT"
bash "$HERE/02_extract.sh"

echo "==== [2/4] filter to XS ===="
rm -rf "$STAGE"
python3 "$HERE/make_tiny_subset.py" \
  --full "$EXTRACT" --out "$STAGE" \
  --split "$SPLIT" --full-split "$FULLSPLIT" --report "$REPORT"

echo "==== [3/4] repackage ===="
rm -rf "$UPLOAD"
bash "$HERE/04_repackage.sh"

echo "==== [4/4] verify (stage content + upload tree) ===="
python3 "$HERE/05_verify.py" --stage "$STAGE" --split "$SPLIT" --upload "$UPLOAD" --check-hf

echo "BUILD COMPLETE. Upload tree: $UPLOAD"
echo "Next (on DTN): bash $HERE/06_upload.sh   then   bash $HERE/07_croissant.sh"
