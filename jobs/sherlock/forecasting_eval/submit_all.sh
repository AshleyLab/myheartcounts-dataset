#!/usr/bin/env bash
# Submit the full forecasting eval suite on Sherlock.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_$(date +%Y%m%d_%H%M%S)}"
export MHC_FORECAST_RUN_LABEL="${RUN_LABEL}"

jobs=(
    jobs/sherlock/forecasting_eval/run_baselines.sbatch
    jobs/sherlock/forecasting_eval/run_chronos2.sbatch
    jobs/sherlock/forecasting_eval/run_toto.sbatch
    jobs/sherlock/forecasting_eval/run_mixlinear.sbatch
    jobs/sherlock/forecasting_eval/run_dlinear.sbatch
    jobs/sherlock/forecasting_eval/run_segrnn.sbatch
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
        jobs/sherlock/forecasting_eval/aggregate_results.sbatch)"
    echo "Submitted aggregation: ${agg_out%%;*}"
fi

echo "RUN_LABEL=${RUN_LABEL}"
