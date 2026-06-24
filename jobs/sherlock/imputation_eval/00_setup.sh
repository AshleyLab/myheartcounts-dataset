#!/usr/bin/env bash
# One-time setup for the imputation-eval pipeline on Sherlock.
#
# Run once on a login node (or via `sh_dev`) BEFORE submitting any sbatch:
#     bash jobs/sherlock/imputation_eval/00_setup.sh
#
# What it does:
#   1. Activates the openmhc venv (somewhere on $SCRATCH; the in-repo
#      activator script handles the path).
#   2. Verifies critical Python imports.
#   3. Downloads all 6 paper W&B artifacts to ${RELEASES}
#      (defaults to ${SCRATCH_RUN_ROOT}/releases).
#   4. Confirms each release bundle has an openmhc_manifest.json sidecar.
#
# Required:
#   - W&B credentials at ~/.netrc (machine api.wandb.ai) OR
#     WANDB_API_KEY env var.
#   - The dataset cache populated under ${MHC_CACHE} (defaults to
#     ${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-full/).
#     If missing, run from a login node:
#         python -c "import openmhc; openmhc.download_dataset(version='full')"

# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

# --- Env smoke test -----------------------------------------------------------
python - <<'PY'
import importlib
mods = ["openmhc", "pypots", "pytorch_lightning", "hydra", "wandb",
        "imputation_evaluation"]
for m in mods:
    importlib.import_module(m)
print("env OK:", ", ".join(mods))
PY

# --- Dataset sanity -----------------------------------------------------------
test -d "$DAILY_HF_DIR" \
  || { echo "[FATAL] daily_hf dir missing: $DAILY_HF_DIR" >&2; exit 1; }
test -f "$SPLIT_FILE" \
  || { echo "[FATAL] split file missing: $SPLIT_FILE" >&2; exit 1; }
test -d "$MASKS/test" -a -d "$MASKS/val" \
  || { echo "[FATAL] masks dir incomplete: $MASKS" >&2; exit 1; }
echo "dataset OK: $DAILY_HF_DIR (split: $(basename "$SPLIT_FILE"))"

# --- W&B artifact downloads ---------------------------------------------------
mkdir -p "$RELEASES"

# Map: release dir name -> W&B artifact identifier
declare -A ARTIFACTS=(
  [brits]="MHC_Dataset/mhc-pypots-brits/brits:v19"
  [dlinear]="MHC_Dataset/mhc-pypots-dlinear/dlinear:v49"
  [fedformer]="MHC_Dataset/mhc-pypots-fedformer/fedformer:v31"
  [timesnet]="MHC_Dataset/mhc-pypots-timesnet/timesnet:v31"
  [lsm2]="MHC_Dataset/mhc-mae-ssl-daily/mae-daily:v0"
  [lsm2_weekly]="MHC_Dataset/mhc-mae-ssl/model-o5quh2cd:v2"
  [lsm2_weekly_sparse]="MHC_Dataset/mhc-mae-ssl/mae-weekly-sparse-d4:v0"
)

for name in "${!ARTIFACTS[@]}"; do
  dir="$RELEASES/$name"
  if [[ -f "$dir/openmhc_manifest.json" ]]; then
    echo "[skip] $name already present at $dir"
    continue
  fi
  echo "[download] $name <- ${ARTIFACTS[$name]}"
  wandb artifact get "${ARTIFACTS[$name]}" --root "$dir"
done

# --- Build manifests ----------------------------------------------------------
# W&B artifacts are bare training dumps; openmhc's release-bundle format expects
# openmhc_manifest.json + normalization_stats.json siblings. Build them now.
python "$(dirname "$0")/_build_manifests.py"

# --- Verify manifests ---------------------------------------------------------
missing=0
for name in "${!ARTIFACTS[@]}"; do
  if [[ ! -f "$RELEASES/$name/openmhc_manifest.json" ]]; then
    echo "[FATAL] missing manifest: $RELEASES/$name/openmhc_manifest.json" >&2
    missing=1
  fi
done
(( missing == 0 )) || exit 1

echo
echo "All 6 paper checkpoint releases verified at $RELEASES/"
ls -la "$RELEASES/"
