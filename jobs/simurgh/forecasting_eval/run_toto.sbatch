#!/usr/bin/env bash
#SBATCH --job-name=fc-toto
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/forecasting_eval/toto_%j.out
#SBATCH --error=logs/forecasting_eval/toto_%j.err

# Toto foundation model. Runs BOTH variants under the shared run label:
#   - toto_zeroshot:  base Datadog/Toto-Open-Base-1.0 (no release_dir)
#   - toto_finetuned: the released fine-tuned bundle (model.release_dir)
# Uses the dedicated openmhc-toto env (has the `toto-ts` package).
# Toto draws num_samples=256 sample paths per window (the long pole).
set -euo pipefail

export MHC_CONDA_ENV=openmhc-toto
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/simurgh/forecasting_eval/_common.sh"

# Toto draws num_samples=256 paths per window. At samples_per_batch=256 the unembed
# activation OOMs the 48 GB L40S (~61 GB peak). Compute the 256 samples in chunks
# instead — same result, bounded peak memory. Measured @2048 context (the model's
# context cap): spb=32 is safe but far too slow (~25h/variant); spb=128 -> 41 GB
# reserved is too tight on the 44 GB L40S; spb=64 -> 21 GB reserved is the
# safe/fast pick (~12-16h/variant). Must divide num_samples (256): use 32/64/128.
TOTO_SAMPLES_PER_BATCH="${TOTO_SAMPLES_PER_BATCH:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
spb="model.toto.samples_per_batch=${TOTO_SAMPLES_PER_BATCH}"

nvidia-smi || true
echo "Start: $(date)  Host: $(hostname)  samples_per_batch=${TOTO_SAMPLES_PER_BATCH}"

ft_dir="${MHC_FORECAST_TOTO_RELEASE_DIR:?set MHC_FORECAST_TOTO_RELEASE_DIR to the fine-tuned bundle}"

echo "=== Running toto (zero-shot, base Datadog/Toto-Open-Base-1.0) ==="
run_forecast_variant toto_zeroshot toto "${spb}" "$@"

echo "=== Running toto (fine-tuned) ==="
run_forecast_variant toto_finetuned toto "model.release_dir=${ft_dir}" "${spb}" "$@"

echo "End: $(date)"
