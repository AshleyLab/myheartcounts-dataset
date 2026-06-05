#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/run_eval_%j.log

# ============================================================================
# Downstream eval — FROM RAW, through the public API. THE single entry point.
#
# Runs openmhc.evaluate_prediction(model) for the bundled baseline named by
# METHOD (the same call an external submitter makes), building features/embeddings
# from raw on a cache miss. One driver for every method — pick it with METHOD.
#
#   METHOD ∈ { linear, multirocket, gru_d,   # CPU
#              toto, chronos2, wbm }          # GPU (first-run extraction); wbm = hybrid SSL
#
# Usage (set MHC_DATA_DIR to the dataset root; add --gres=gpu:1 for GPU methods):
#   METHOD=linear   MHC_DATA_DIR=/path/to/data sbatch               jobs/imperial/slurm/run_eval.slurm
#   METHOD=toto     MHC_DATA_DIR=/path/to/data sbatch --gres=gpu:1  jobs/imperial/slurm/run_eval.slurm
#
# Output: $OUT_CSV (default eval_<METHOD>.csv). Results are also returned in-process
# as a PredictionResults object (.to_csv / .to_json / .summary) by the public API.
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
