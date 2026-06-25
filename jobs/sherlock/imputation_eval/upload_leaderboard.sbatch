#!/bin/bash
# Upload all 17 per-method substrate parquets + TWO bootstrap references
# to MyHeartCounts/OpenMHC-leaderboard-data on Hugging Face.
#
# Inputs:
#   /scratch/.../paper-verification/per_user/<method>.parquet   (17 substrates, from producer array)
#   /scratch/.../paper/bootstrap_draws.parquet                  (17-method, with lsm2_weekly)
#   /scratch/.../paper_no_dense/bootstrap_draws.parquet         (16-method, no lsm2_weekly)
#
# HF layout after this runs:
#   imputation/<method>.parquet                                 (16 updated + 1 new = 17)
#   imputation/bootstrap/draws.parquet                          ← 16-method (REPLACES existing)
#   imputation/bootstrap_with_dense_weekly/draws.parquet        ← 17-method (NEW sibling)
#
# All 17 methods get a sidecar (meta.json) write because every upload passes
# --results-json, which auto-extracts fallback_rate. The tool fetches each
# method's existing HF sidecar first and merges in only the provided fields, so
# display_name/type/submitter/subtrack are preserved verbatim for the 16
# already-on-HF methods. lsm2_weekly is the only new method — we ship its full
# display fields on top of the auto-extracted fallback_rate.
#
# Auth: reads ~/.cache/huggingface/token and exports as HF_TOKEN (works around
# huggingface_hub 1.4.1 default HF_HOME pointing to /tmp/huggingface).

#SBATCH --job-name=mhc-leaderboard-upload
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/%u/logs/openmhc/upload_leaderboard_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/upload_leaderboard_%j.err

set -euo pipefail

# Use absolute path: sbatch copies the script to /var/spool/slurmd at run
# time, so $BASH_SOURCE doesn't point at the in-repo sibling _common.sh.
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

# HF auth: pick up cached token explicitly.
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
echo "[upload] HF_TOKEN length=${#HF_TOKEN}"

PER_USER_DIR="${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper-verification/per_user"
PAPER_DIR="${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper"
PAPER_NO_DENSE_DIR="${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper_no_dense"
REPO=MyHeartCounts/OpenMHC-leaderboard-data

METHODS=(
  mean mode locf linear temporal_mean temporal_mode
  personalized_mean personalized_mode personalized_temporal_mean
  brits dlinear dlinear_weekly fedformer timesnet
  lsm2 lsm2_weekly_sparse lsm2_weekly
)
RUNS="${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/runs"

echo
echo "==== 1) Upload per-method substrate parquets (17 methods) ===="
# The tool auto-extracts fallback_rate from --results-json and merges it into
# the HF sidecar without clobbering display_name/type/submitter/subtrack.
# The 16 already-on-HF methods get only --results-json (so the renderer-side
# display fields they already have on HF are preserved verbatim).
# lsm2_weekly is the only new-on-HF method, so we ship its display fields
# explicitly on the first upload; re-passing them on subsequent runs is
# idempotent because the tool merges into the existing sidecar.
for M in "${METHODS[@]}"; do
  src="${PER_USER_DIR}/${M}.parquet"
  if [[ ! -f "$src" ]]; then
    echo "[upload] SKIP $M: $src missing"
    continue
  fi
  base_args=(
    --dir "$PER_USER_DIR"
    --method "$M"
    --track imputation
    --no-private
    --results-json "${RUNS}/${M}/results.json"
  )
  if [[ "$M" == "lsm2_weekly" ]]; then
    python tools/upload_leaderboard_substrate.py "${base_args[@]}" \
      --name "LSM-2 (7-day)" \
      --type Neural \
      --submitter "OpenMHC team" \
      --subtrack long-context
  else
    python tools/upload_leaderboard_substrate.py "${base_args[@]}"
  fi
done

echo
echo "==== 2a) Upload 16-method bootstrap to imputation/bootstrap/ (replaces existing) ===="
python tools/upload_leaderboard_bootstrap.py \
  --dir "$PAPER_NO_DENSE_DIR" \
  --track imputation \
  --no-private

echo
echo "==== 2b) Upload 17-method bootstrap to imputation/bootstrap_with_dense_weekly/ (new sibling) ===="
python - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
src_dir = Path("${PAPER_DIR}")
# Files: bootstrap_draws.parquet (required), bootstrap_draws.parquet.meta.json (optional)
files = [
    ("bootstrap_draws.parquet", "draws.parquet", True),
    ("bootstrap_draws.parquet.meta.json", "draws.meta.json", False),
]
dest_prefix = "imputation/bootstrap_with_dense_weekly"
for src_name, dest_name, required in files:
    src = src_dir / src_name
    if not src.exists():
        print(f"  {src_name}: MISSING (required={required})")
        if required:
            raise SystemExit(f"  -> ABORT, required file missing")
        continue
    dest = f"{dest_prefix}/{dest_name}"
    size_mb = src.stat().st_size / 1024 / 1024
    print(f"  uploading {src} ({size_mb:.1f} MB) -> {dest}")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=dest,
        repo_id="MyHeartCounts/OpenMHC-leaderboard-data",
        repo_type="dataset",
        commit_message=f"Add 17-method (with lsm2_weekly dense) bootstrap: {dest_name}",
    )
# 17-method bootstrap docs live in their own sibling subdir (tells readers
# the method pool has 17 entries instead of 16, and that rank ranges to 17).
docs_dir = Path("/home/users/schuetzn/myheartcounts-dataset/tools/leaderboard_docs/imputation/bootstrap_with_dense_weekly")
if docs_dir.is_dir():
    for doc in sorted(docs_dir.glob("*.md")):
        dest = f"{dest_prefix}/{doc.name}"
        print(f"  uploading {doc.name} -> {dest}")
        api.upload_file(
            path_or_fileobj=str(doc),
            path_in_repo=dest,
            repo_id="MyHeartCounts/OpenMHC-leaderboard-data",
            repo_type="dataset",
            commit_message=f"Sync docs for {dest_prefix}",
        )
PY

echo
echo "==== 2c) Upload top-level imputation/ track docs ===="
# Per-method substrate README/SCHEMA. Documents the imputation/<m>.parquet
# columns and the imputation/<m>.meta.json sidecar (incl. fallback_rate).
python - <<PY
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
docs_dir = Path("/home/users/schuetzn/myheartcounts-dataset/tools/leaderboard_docs/imputation")
for doc_name in ("README.md", "SCHEMA.md"):
    doc = docs_dir / doc_name
    if not doc.exists():
        print(f"  {doc_name}: MISSING — skipping")
        continue
    dest = f"imputation/{doc_name}"
    print(f"  uploading {doc_name} -> {dest}")
    api.upload_file(
        path_or_fileobj=str(doc),
        path_in_repo=dest,
        repo_id="MyHeartCounts/OpenMHC-leaderboard-data",
        repo_type="dataset",
        commit_message=f"Sync imputation/{doc_name}",
    )
PY

echo
echo "==== 3) Final listing of imputation/ on HF ===="
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
files = sorted(api.list_repo_files("MyHeartCounts/OpenMHC-leaderboard-data", repo_type="dataset"))
print(f"Repo now has {len(files)} files. Imputation track:")
for f in files:
    if f.startswith("imputation/"):
        print(" ", f)
PY
