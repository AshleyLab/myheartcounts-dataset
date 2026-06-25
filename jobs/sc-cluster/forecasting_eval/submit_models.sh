#!/usr/bin/env bash
# Submit the GPU forecasting *model* suite on Simurgh (SC), to land alongside the
# CPU baselines (submit_all.sh) under a shared run label.
#
# 5 jobs / 7 evaluations:
#   run_dlinear / run_segrnn / run_mixlinear  -> fine-tuned neural (openmhc env)
#   run_chronos2 -> chronos2_zeroshot + chronos2_finetuned   (openmhc-chronos2)
#   run_toto     -> toto_zeroshot     + toto_finetuned       (openmhc-toto)
#
# By default it JOINS the in-flight baseline run label so a single aggregation
# covers all models. Override with MHC_FORECAST_RUN_LABEL.
#
# Fine-tuned bundles default to ${REPO}/releases-fc/openmhc-<model>-fc
# (downloaded from MyHeartCounts/openmhc-<model>-fc). Override the dir with
# MHC_FORECAST_RELEASES_DIR or each MHC_FORECAST_<MODEL>_RELEASE_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

# To aggregate GPU models together with the CPU baselines, export the SAME
# MHC_FORECAST_RUN_LABEL you used for submit_all.sh (or just use submit_pipeline.sh,
# which fans out everything under one label). Standalone, this defaults to a FRESH
# date label — never a pinned historical run — so a bare invocation can't silently
# append to / overwrite an old (pre-fix) eval.
RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_$(date +%Y%m%d_%H%M%S)}"
export MHC_FORECAST_RUN_LABEL="${RUN_LABEL}"

# Local release bundles.
RELEASES="${MHC_FORECAST_RELEASES_DIR:-${REPO_DIR}/releases-fc}"
export MHC_FORECAST_DLINEAR_RELEASE_DIR="${MHC_FORECAST_DLINEAR_RELEASE_DIR:-${RELEASES}/openmhc-dlinear-fc}"
export MHC_FORECAST_SEGRNN_RELEASE_DIR="${MHC_FORECAST_SEGRNN_RELEASE_DIR:-${RELEASES}/openmhc-segrnn-fc}"
export MHC_FORECAST_MIXLINEAR_RELEASE_DIR="${MHC_FORECAST_MIXLINEAR_RELEASE_DIR:-${RELEASES}/openmhc-mixlinear-fc}"
export MHC_FORECAST_CHRONOS2_RELEASE_DIR="${MHC_FORECAST_CHRONOS2_RELEASE_DIR:-${RELEASES}/openmhc-chronos2-fc}"
export MHC_FORECAST_TOTO_RELEASE_DIR="${MHC_FORECAST_TOTO_RELEASE_DIR:-${RELEASES}/openmhc-toto-fc}"

submitted=()

# The 3 neural models share ONE history_cf cache bundle. On a cold cache,
# concurrent first-time builds race on the HDF5 file lock (BlockingIOError).
# Serialize them: the first builds the cache, the other two start afterok and
# reuse it. On a warm cache they all just reuse the existing bundle.
first="$(sbatch --parsable --export=ALL jobs/sc-cluster/forecasting_eval/run_dlinear.sbatch "$@")"
first="${first%%;*}"
submitted+=("${first}")
echo "Submitted run_dlinear.sbatch: ${first}"
for job in run_segrnn.sbatch run_mixlinear.sbatch; do
    out="$(sbatch --parsable --dependency="afterok:${first}" --export=ALL "jobs/sc-cluster/forecasting_eval/${job}" "$@")"
    job_id="${out%%;*}"
    submitted+=("${job_id}")
    echo "Submitted ${job}: ${job_id} (afterok:${first})"
done

# Foundation models use independent per-model caches -> safe to run in parallel.
for job in run_chronos2.sbatch run_toto.sbatch; do
    out="$(sbatch --parsable --export=ALL "jobs/sc-cluster/forecasting_eval/${job}" "$@")"
    job_id="${out%%;*}"
    submitted+=("${job_id}")
    echo "Submitted ${job}: ${job_id}"
done

# Chain a fresh aggregation (afterok) over whatever metrics exist for this label.
# NOTE: if autoARIMA is still running it won't be in this summary — just re-run
# aggregate_results.sbatch once it finishes for the final all-models table.
if [[ "${MHC_FORECAST_AGGREGATE:-1}" == "1" ]]; then
    dependency="$(IFS=:; echo "${submitted[*]}")"
    agg_out="$(sbatch --parsable \
        --dependency="afterok:${dependency}" \
        --export=ALL \
        jobs/sc-cluster/forecasting_eval/aggregate_results.sbatch)"
    echo "Submitted aggregation: ${agg_out%%;*}"
fi

echo "RUN_LABEL=${RUN_LABEL}"
