#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/run_eval_%j.log

# ============================================================================
# Generic downstream eval — FROM RAW, through the public API.
#
# Runs openmhc.evaluate_prediction(model) for the bundled baseline named by
# METHOD (the same call an external submitter makes). One engine, build-on-miss
# inside the model on a cache miss. GPU methods (toto/chronos2/wbm) need a GPU:
# add `--gres=gpu:1` on the sbatch line. CPU methods (linear/multirocket) don't.
#
# Usage:
#   METHOD=toto        MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch --gres=gpu:1 jobs/imperial/slurm/run_eval.slurm
#   METHOD=multirocket MHC_DATA_DIR=/home/yb525/yuze_mhc_ws/data sbatch              jobs/imperial/slurm/run_eval.slurm
# ============================================================================

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$REPO/logs"
cd "$REPO"

export METHOD="${METHOD:-linear}"
echo "=== downstream eval via openmhc.evaluate_prediction: METHOD=$METHOD ==="
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

# Model artifacts for the foundation models (external deps). For validation these
# point at the private copies; public release sets them to the submodule + HF/W&B refs.
PRIV="${PRIV:-/home/yb525/yuze_mhc_ws}"
export TOTO_REPO="${TOTO_REPO:-$PRIV/external/toto}"
export TOTO_CHECKPOINT="${TOTO_CHECKPOINT:-$PRIV/models/final/toto_FT/toto-epoch=24-step=116225-val_loss=-1.3597.ckpt}"
export CHRONOS_REPO="${CHRONOS_REPO:-$PRIV/external/chronos-forecasting}"
export CHRONOS2_CHECKPOINT="${CHRONOS2_CHECKPOINT:-$PRIV/models/final/chronos2_FT}"

export OUT_CSV="${OUT_CSV:-eval_${METHOD}.csv}"

python jobs/imperial/slurm/run_eval.py

echo "=== downstream eval complete ($METHOD) -> $OUT_CSV ==="
date
