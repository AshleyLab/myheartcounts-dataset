#!/bin/bash
# Train FEDformer from scratch with deterministic seeding + FourierBlock
# sidecar capture, producing a release bundle that loads byte-identical
# across processes (fixing the upstream PyPOTS bug).
#
# Hyperparameters match the published paper checkpoint exactly —
# W&B run MHC_Dataset/mhc-pypots-fedformer/runs/ouqezdi7
# (artifact ``fedformer:v31``). See
# configs/training/training/fedformer_paper.yaml for the source of truth.
#
# Resource budget: 48h is the Sherlock ``gpu`` partition cap; the paper
# run on Stanford simurgh used 64G mem + 1 GPU.
#
# Expected output: a release bundle at
#   $RELEASES_ROOT/fedformer_<timestamp>/
# containing model.pypots, normalization_stats.json, fourier_modes.json,
# and openmhc_manifest.json — directly consumable by the eval CLI via
#   mhc-impute-eval method=fedformer method.release_dir=<release_dir>

#SBATCH --job-name=mhc-impute-train-fedformer
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
# Tesla-family GPUs only — consumer RTX 3090s on Sherlock occasionally
# throw "GPU is lost" NVML errors that cause silent SLURM requeues
# (see jobs/sherlock/imputation_eval/run_timesnet.sbatch for context).
#SBATCH --constraint=GPU_BRD:TESLA&(GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB)
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/train_fedformer_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/train_fedformer_%j.err

# shellcheck disable=SC1091
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_train/_common.sh

nvidia-smi || true

TS=$(date +%Y%m%d_%H%M%S)
SAVING_PATH="${RUNS_ROOT}/fedformer_${TS}"
RELEASE_DIR="${RELEASES_ROOT}/fedformer_${TS}"

mhc-impute-train \
  model=fedformer \
  training=fedformer_paper \
  seed=42 \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  +data.split_file="${SPLIT_FILE}" \
  data.batch_size=500 \
  data.num_workers=4 \
  h5_export.output_dir="${H5_CACHE}" \
  output.saving_path="${SAVING_PATH}" \
  output.release_dir="${RELEASE_DIR}"

echo
echo "Release bundle ready at: ${RELEASE_DIR}"
echo
echo "To re-run eval against this checkpoint:"
echo "  Edit jobs/sherlock/imputation_eval/run_fedformer.sbatch and change"
echo "  method.release_dir=\${RELEASES}/fedformer  →  method.release_dir=${RELEASE_DIR}"
echo "  Then resubmit:"
echo "    sbatch jobs/sherlock/imputation_eval/run_fedformer.sbatch"
echo "  And after that:"
echo "    python jobs/sherlock/imputation_eval/verify_parity.py --methods fedformer"
echo "  Expect: all FEDformer rows now within strict 1% tolerance."
