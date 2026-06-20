#!/usr/bin/env bash
#SBATCH --job-name=fc-aggregate
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/forecasting_eval/aggregate_%j.out
#SBATCH --error=logs/forecasting_eval/aggregate_%j.err

# DEPRECATED — superseded by jobs/sc-cluster/forecasting_eval/skill_rank.sbatch.
# Runs the (deprecated) MAE-by-channel-hour aggregator. Kept only to reproduce
# the old mae_by_model_channel_hour.csv / statistical_result.csv tables.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

summary_dir="${RUNS_ROOT}/summary/${RUN_LABEL}"
mkdir -p "${summary_dir}"

args=()
for metrics_dir in "${RUNS_ROOT}"/*/predictions/"${RUN_LABEL}"/*_metrics/"${RUN_LABEL}"; do
    [[ -d "${metrics_dir}" ]] || continue
    model="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "${metrics_dir}")")")")")"
    args+=(--model "${model}=${metrics_dir}")
done

if [[ "${#args[@]}" -eq 0 ]]; then
    echo "No metrics directories found for RUN_LABEL=${RUN_LABEL} under ${RUNS_ROOT}" >&2
    exit 1
fi

python src/forecasting_evaluation/metrics/deprecated/summary_metrics_result.py \
    "${args[@]}" \
    --output-dir "${summary_dir}"
