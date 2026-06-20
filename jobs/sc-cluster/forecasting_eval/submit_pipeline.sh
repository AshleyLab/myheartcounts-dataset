#!/usr/bin/env bash
# One-command end-to-end forecasting paper run on the SC cluster.
#
# Fans out every per-model eval job under a single RUN_LABEL (SLURM), then chains
# the paper pipeline (discover + skill + rank) via an afterok dependency on all of
# them. Mirrors jobs/sherlock/imputation_eval/submit_all.sh.
#
# Usage:
#   jobs/sc-cluster/forecasting_eval/submit_pipeline.sh            # full set + pipeline
#   MHC_FORECAST_RUN_LABEL=myrun jobs/.../submit_pipeline.sh    # pin the label
#   MHC_FORECAST_PIPELINE=0 jobs/.../submit_pipeline.sh         # eval jobs only, no pipeline
#   jobs/.../submit_pipeline.sh --only dlinear segrnn           # subset of models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs/forecasting_eval

RUN_LABEL="${MHC_FORECAST_RUN_LABEL:-forecasting_paper_$(date +%Y%m%d_%H%M%S)}"
export MHC_FORECAST_RUN_LABEL="${RUN_LABEL}"

# Release bundles (override via env). Retrained pypots + published foundation bundles.
RELEASES="${MHC_FORECAST_RELEASES_DIR:-${REPO_DIR}/releases-fc}"
TRAIN_REL="${REPO_DIR}/results/forecasting_train/sc-cluster/releases"
export MHC_FORECAST_DLINEAR_RELEASE_DIR="${MHC_FORECAST_DLINEAR_RELEASE_DIR:-${TRAIN_REL}/dlinear_20260609_024228}"
export MHC_FORECAST_MIXLINEAR_RELEASE_DIR="${MHC_FORECAST_MIXLINEAR_RELEASE_DIR:-${TRAIN_REL}/mixlinear_20260609_025728}"
export MHC_FORECAST_SEGRNN_RELEASE_DIR="${MHC_FORECAST_SEGRNN_RELEASE_DIR:-${TRAIN_REL}/segrnn_20260609_031230}"
export MHC_FORECAST_CHRONOS2_RELEASE_DIR="${MHC_FORECAST_CHRONOS2_RELEASE_DIR:-${RELEASES}/openmhc-chronos2-fc}"
export MHC_FORECAST_TOTO_RELEASE_DIR="${MHC_FORECAST_TOTO_RELEASE_DIR:-${RELEASES}/openmhc-toto-fc}"

# Optional model filter: --only <names...>
declare -a ONLY=()
while (( $# )); do
  case "$1" in
    --only) shift; while (( $# )) && [[ "$1" != --* ]]; do ONLY+=("$1"); shift; done ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
want() { (( ${#ONLY[@]} == 0 )) && return 0; local x; for x in "${ONLY[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

MANIFEST="${REPO_DIR}/results/forecasting_eval/sc-cluster/${RUN_LABEL}_job_manifest.tsv"
mkdir -p "$(dirname "${MANIFEST}")"
: > "${MANIFEST}"
printf '# jobid\tlabel\tscript\t(%s)\n' "$(date -Is)" >> "${MANIFEST}"

# submit <label> <script> [extra sbatch args...] -> prints jobid (stdout), logs to stderr.
submit() {
  local label=$1 script=$2; shift 2
  local jid
  jid=$(sbatch --parsable --exclude=simurgh2 \
        --export=ALL,MHC_FORECAST_RUN_LABEL="${RUN_LABEL}" \
        "$@" "jobs/sc-cluster/forecasting_eval/${script}")
  jid=${jid%%;*}
  printf '%s\t%s\t%s\n' "${jid}" "${label}" "${script}" >> "${MANIFEST}"
  printf '[submit] %-22s %s (%s)\n' "${label}" "${jid}" "${script}" >&2
  printf '%s\n' "${jid}"
}

ids=()
# CPU baselines
want naive     && ids+=("$(submit naive     run_naive.sbatch)")
want autoets   && ids+=("$(submit autoets   run_autoets.sbatch)")
want autoarima && ids+=("$(submit autoarima run_autoarima.sbatch)")
# Foundation models (GPU, independent)
want chronos2  && ids+=("$(submit chronos2  run_chronos2.sbatch)")
want toto      && ids+=("$(submit toto      run_toto.sbatch)")
# PyPOTS (GPU): dlinear builds the shared history_cf cache; segrnn/mixlinear afterok-chain it.
dl=""
if want dlinear; then dl="$(submit dlinear run_dlinear.sbatch)"; ids+=("${dl}"); fi
dep=(); [[ -n "${dl}" ]] && dep=(--dependency="afterok:${dl}")
want segrnn    && ids+=("$(submit segrnn    run_segrnn.sbatch    "${dep[@]}")")
want mixlinear && ids+=("$(submit mixlinear run_mixlinear.sbatch "${dep[@]}")")

# Chain the paper pipeline (discover + skill + rank) after ALL eval jobs.
if [[ "${MHC_FORECAST_PIPELINE:-1}" == "1" && ${#ids[@]} -gt 0 ]]; then
  DEP="afterok:$(IFS=:; echo "${ids[*]}")"
  pj=$(sbatch --parsable --exclude=simurgh2 \
        --export=ALL,MHC_FORECAST_RUN_LABEL="${RUN_LABEL}" \
        --dependency="${DEP}" \
        jobs/sc-cluster/forecasting_eval/run_paper_pipeline.sbatch)
  pj=${pj%%;*}
  printf '%s\t%s\t%s\n' "${pj}" "paper_pipeline" "run_paper_pipeline.sbatch" >> "${MANIFEST}"
  echo "[submit] paper_pipeline ${pj} (afterok on ${#ids[@]} eval jobs)"
fi

echo
echo "RUN_LABEL=${RUN_LABEL}"
echo "Submitted ${#ids[@]} eval jobs. Manifest: ${MANIFEST}"
echo "Results will land in: results/forecasting_eval/sc-cluster/summary/${RUN_LABEL}/"
