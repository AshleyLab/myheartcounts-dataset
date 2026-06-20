#!/usr/bin/env bash
# Submit the CPU forecasting-baseline suite on the SC cluster.
#
# Submits the 3 CPU jobs (naive / autoETS / autoARIMA) under a shared run label,
# then chains the summary aggregation with an afterok dependency (set
# MHC_FORECAST_AGGREGATE=0 to skip).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_$(date +%Y%m%d_%H%M%S)}"
export MHC_FORECAST_RUN_LABEL="${RUN_LABEL}"

jobs=(
    jobs/sc-cluster/forecasting_eval/run_naive.sbatch
    jobs/sc-cluster/forecasting_eval/run_autoets.sbatch
    jobs/sc-cluster/forecasting_eval/run_autoarima.sbatch
)

submitted=()
for job in "${jobs[@]}"; do
    out="$(sbatch --parsable --export=ALL,MHC_FORECAST_RUN_LABEL="${RUN_LABEL}" "${job}" "$@")"
    job_id="${out%%;*}"
    submitted+=("${job_id}")
    echo "Submitted ${job}: ${job_id}"
done

if [[ "${MHC_FORECAST_AGGREGATE:-1}" == "1" ]]; then
    dependency="$(IFS=:; echo "${submitted[*]}")"
    agg_out="$(sbatch --parsable \
        --dependency="afterok:${dependency}" \
        --export=ALL,MHC_FORECAST_RUN_LABEL="${RUN_LABEL}" \
        jobs/sc-cluster/forecasting_eval/aggregate_results.sbatch)"
    echo "Submitted aggregation: ${agg_out%%;*}"
fi

echo "RUN_LABEL=${RUN_LABEL}"
