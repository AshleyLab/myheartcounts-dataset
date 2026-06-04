#!/bin/bash
#SBATCH --job-name=run_wbm
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_wbm_%j.log

# ============================================================================
# WBM downstream eval — single flow (FROM RAW), GPU.
#
# Runs run_eval(WBM()): on a cache miss the WBM model extracts its embeddings
# from raw (Mamba2 encoder over weekly tensors), saves them, and reuses them on
# a hit — all inside one eval flow — then the engine runs PCA-50 + the uniform
# probe. First run is genuinely from raw; later runs are fast (warm cache).
#
# (Temporary jobs/ scaffolding for our validation — remove before release.)
#
# Usage:
#   MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch jobs/imperial/slurm/run_wbm.slurm
# ============================================================================

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$REPO/logs"
cd "$REPO"

echo "=== WBM eval (single from-raw flow) ==="
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
export HF_HOME="$TMPDIR/hf_cache"
mkdir -p "$HF_HOME"

: "${MHC_DATA_DIR:?set MHC_DATA_DIR to the dataset root}"
export OUT_CSV="${OUT_CSV:-eval_wbm.csv}"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python jobs/imperial/slurm/run_wbm_eval.py

echo "=== WBM eval complete -> $OUT_CSV ==="
date
