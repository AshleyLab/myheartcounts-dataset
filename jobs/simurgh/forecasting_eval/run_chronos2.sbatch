#!/usr/bin/env bash
#SBATCH --job-name=fc-chronos2
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/forecasting_eval/chronos2_%j.out
#SBATCH --error=logs/forecasting_eval/chronos2_%j.err

# Chronos-2 foundation model. Runs BOTH variants under the shared run label:
#   - chronos2_zeroshot:  base amazon/chronos-2 (no release_dir)
#   - chronos2_finetuned: the released fine-tuned bundle (model.release_dir)
# Uses the dedicated openmhc-chronos2 env (has the `chronos` package).
set -euo pipefail

export MHC_CONDA_ENV=openmhc-chronos2
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/simurgh/forecasting_eval/_common.sh"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)"

ft_dir="${MHC_FORECAST_CHRONOS2_RELEASE_DIR:?set MHC_FORECAST_CHRONOS2_RELEASE_DIR to the fine-tuned bundle}"

echo "=== Running chronos2 (zero-shot, base amazon/chronos-2) ==="
run_forecast_variant chronos2_zeroshot chronos2 "$@"

echo "=== Running chronos2 (fine-tuned) ==="
run_forecast_variant chronos2_finetuned chronos2 "model.release_dir=${ft_dir}" "$@"

echo "End: $(date)"
