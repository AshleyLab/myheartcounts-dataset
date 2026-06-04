#!/bin/bash
#SBATCH --job-name=run_tsfm
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_tsfm_%j.log

# ============================================================================
# TSFM encoder downstream eval — single flow (FROM RAW), GPU.
#
# Runs run_eval(Toto()/Chronos2()): on a cache miss the model extracts its
# per-(split, task) last-latent embeddings from raw (daily_hourly_hf → label-
# aligned 2048h windows → backbone), saves the HDF5s, reuses them on a hit — all
# inside one eval flow — then the engine channel-mean-pools + runs PCA-50 + probe.
#
# The pretrained model is an external dependency resolved by reference; for our
# validation the *_REPO / *_CHECKPOINT envs point at the private artifacts.
#
# (Temporary jobs/ scaffolding for validation — remove before release.)
#
# Usage:
#   METHOD=toto      MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch jobs/imperial/slurm/run_tsfm.slurm
#   METHOD=chronos2  MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch jobs/imperial/slurm/run_tsfm.slurm
# ============================================================================

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$REPO/logs"
cd "$REPO"

export METHOD="${METHOD:-toto}"
echo "=== TSFM eval (single from-raw flow): METHOD=$METHOD ==="
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
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

: "${MHC_DATA_DIR:?set MHC_DATA_DIR to the dataset root}"

# Model artifacts (external deps). For validation these point at the private repo;
# public release sets them to the submodule checkout + HF/W&B refs.
PRIV="${PRIV:-/home/yb525/yuze_mhc_ws}"
export TOTO_REPO="${TOTO_REPO:-$PRIV/external/toto}"
export TOTO_CHECKPOINT="${TOTO_CHECKPOINT:-$PRIV/models/final/toto_FT/toto-epoch=24-step=116225-val_loss=-1.3597.ckpt}"
export CHRONOS_REPO="${CHRONOS_REPO:-$PRIV/external/chronos-forecasting}"
export CHRONOS2_CHECKPOINT="${CHRONOS2_CHECKPOINT:-$PRIV/models/final/chronos2_FT}"

export OUT_CSV="${OUT_CSV:-eval_${METHOD}.csv}"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python jobs/imperial/slurm/run_tsfm_eval.py

echo "=== TSFM eval complete ($METHOD) -> $OUT_CSV ==="
date
