#!/usr/bin/env bash
# Submit every imputation-eval job + the paper-bootstrap job that depends
# on all of them. Writes a TSV manifest (jobid \t label \t script) consumed
# by babysit.sh.
#
# Usage:
#   bash jobs/sherlock/imputation_eval/submit_all.sh
#   bash jobs/sherlock/imputation_eval/submit_all.sh --no-paper   # skip paper-bootstrap
#   bash jobs/sherlock/imputation_eval/submit_all.sh --only brits dlinear

set -euo pipefail

JOBS_DIR=/home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval
OUT_BASE=/scratch/users/schuetzn/openmhc-imputation-eval
MANIFEST=${OUT_BASE}/job_manifest.tsv
mkdir -p "$OUT_BASE" /scratch/users/schuetzn/logs/openmhc

# --- args ----------------------------------------------------------------
RUN_PAPER=1
declare -a ONLY=()
while (( $# )); do
  case "$1" in
    --no-paper) RUN_PAPER=0; shift ;;
    --only)
      shift
      while (( $# )) && [[ "$1" != --* ]]; do ONLY+=("$1"); shift; done ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

want() {
  (( ${#ONLY[@]} == 0 )) && return 0
  local x; for x in "${ONLY[@]}"; do [[ "$x" == "$1" ]] && return 0; done
  return 1
}

# --- manifest setup ------------------------------------------------------
: > "$MANIFEST"
echo "# jobid<TAB>label<TAB>script   (written $(date -Is))" >> "$MANIFEST"

submit() {
  # Writes the manifest row to $MANIFEST and prints ONLY the jobid to stdout
  # (so `jid=$(submit ...)` captures cleanly).
  local label=$1 script=$2; shift 2
  local jid
  jid=$(sbatch --parsable "$@" "${JOBS_DIR}/${script}")
  printf "%s\t%s\t%s\n" "$jid" "$label" "$script" >> "$MANIFEST"
  printf "[submit] %-20s %s (script=%s)\n" "$label" "$jid" "$script" >&2
  printf "%s\n" "$jid"
}

# --- imputer jobs --------------------------------------------------------
declare -A JOBS=(
  [baselines]=run_baselines.sbatch
  [brits]=run_brits.sbatch
  [dlinear]=run_dlinear.sbatch
  [dlinear_weekly]=run_dlinear_weekly.sbatch
  [fedformer]=run_fedformer.sbatch
  [timesnet]=run_timesnet.sbatch
  [lsm2]=run_lsm2.sbatch
  [lsm2_weekly_sparse]=run_lsm2_weekly_sparse.sbatch
)
ORDER=(baselines brits dlinear dlinear_weekly fedformer timesnet lsm2 lsm2_weekly_sparse)

IDS=()
for label in "${ORDER[@]}"; do
  want "$label" || { echo "[skip] $label (not in --only filter)"; continue; }
  jid=$(submit "$label" "${JOBS[$label]}")
  IDS+=("$jid")
done

# --- paper-bootstrap (afterok dependency on all imputer jobs) -----------
if (( RUN_PAPER )) && (( ${#IDS[@]} > 0 )); then
  DEP="afterok:$(IFS=:; echo "${IDS[*]}")"
  submit paper_bootstrap run_paper_bootstrap.sbatch --dependency="$DEP" >/dev/null
elif (( RUN_PAPER )); then
  echo "[skip] paper_bootstrap (no imputer jobs to depend on)"
fi

echo
n=$(grep -vc '^#' "$MANIFEST")
echo "Submitted $n jobs. Manifest: $MANIFEST"
echo "Monitor with: squeue -u \$USER"
echo "Babysit hourly with:"
echo "  /loop 1h bash ${JOBS_DIR}/babysit.sh"
