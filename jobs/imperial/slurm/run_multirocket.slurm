#!/bin/bash
#SBATCH --job-name=run_multirocket
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/run_multirocket_%j.log

# ============================================================================
# MultiRocket downstream eval — single flow (FROM RAW), CPU.
#
# Runs run_eval(MultiRocket()): on a cache miss the model fits MultiRocket kernels
# on the train segments, chunk-transforms all ~1.1M daily segments (the full
# (N, 49728) matrix is never materialised), and pools per-(task, user) over each
# task's before-label segments — then the engine runs PCA-50 + the uniform probe.
# Deterministic (random_state=42). No GPU. High RAM for the per-(task,user) pool.
#
# (Temporary jobs/ scaffolding for validation — remove before release.)
#
# Usage:
#   MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch jobs/imperial/slurm/run_multirocket.slurm
# ============================================================================

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$REPO/logs"
cd "$REPO"

echo "=== MultiRocket eval (single from-raw flow, CPU) ==="
echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local} | Date: $(date)"

# -- Conda (source ~/.bashrc BEFORE conda activate) --
set +u
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py310
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TMPDIR="/tmp/slurm_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

: "${MHC_DATA_DIR:?set MHC_DATA_DIR to the dataset root}"
export OUT_CSV="${OUT_CSV:-eval_multirocket.csv}"

python jobs/imperial/slurm/run_multirocket_eval.py

echo "=== MultiRocket eval complete -> $OUT_CSV ==="
date
