#!/bin/bash
# Run all 9 CPU baseline imputers sequentially in one SLURM job.
# Each method writes to ${RUNS_ROOT}/<method>/ so the paper-bootstrap
# pipeline can pick them up by name from the sweep config.
#
# Resources mirror MHC-benchmark/.../run_imputation_baselines_full.sbatch
# (normal partition, 12 CPUs, 128G — needed by personalized variants).

#SBATCH --job-name=mhc-impute-baselines
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/scratch/users/%u/logs/openmhc/baselines_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/baselines_%j.err

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

METHODS=(
  mean
  mode
  linear
  locf
  temporal_mean
  temporal_mode
  personalized_mean
  personalized_mode
  personalized_temporal_mean
)

for M in "${METHODS[@]}"; do
  echo "==== $(date -Is)  running method=$M ===="
  mhc-impute-eval \
    method="$M" \
    hydra.run.dir="${RUNS_ROOT}/${M}" \
    output.results_dir="${RUNS_ROOT}/${M}" \
    data.version=full \
    data.daily_hf_dir="${DAILY_HF_DIR}" \
    data.split_file="${SPLIT_FILE}" \
    +masking.masks_file="${MASKS}" \
    data.num_eval_workers=6 \
    data.num_workers=8 \
    seed=42
done

echo "==== $(date -Is)  baselines done ===="
