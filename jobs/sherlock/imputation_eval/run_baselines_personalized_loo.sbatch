#!/bin/bash
# Re-run only the three personalized baselines after the LOSO leakage fix
# (src/openmhc/imputers/_personalized_base.py). The non-personalized baselines
# (mean, mode, linear, locf, temporal_mean, temporal_mode) are unchanged and
# do not need rerunning.
#
# Each method writes to ${RUNS_ROOT}/<method>/ — overwriting the prior leaky
# results. Downstream paper pipeline + leaderboard JSON build picks up the new
# CSVs by method name (see scripts/paper_results/run_paper_pipeline.py and
# scripts/paper_results/build_leaderboard_json.py).
#
# Resources mirror run_baselines.sbatch (normal, 12 CPUs, 128G). Walltime is
# tighter since we run 3/9 methods.

#SBATCH --job-name=mhc-impute-personalized-loo
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/%u/logs/openmhc/personalized_loo_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/personalized_loo_%j.err

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

METHODS=(
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
    evaluation.save_pairs=true \
    bootstrap=on \
    seed=42
done

echo "==== $(date -Is)  personalized LOSO rerun done ===="
