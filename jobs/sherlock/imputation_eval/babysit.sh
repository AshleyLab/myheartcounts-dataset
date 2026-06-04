#!/usr/bin/env bash
# Hourly watchdog for the imputation-eval batch.
# Reads $MANIFEST, queries sacct for each (latest) jobid per label, resubmits
# failed/timed-out/OOM/cancelled jobs (with the paper_bootstrap afterok
# dependency rewritten to point at the live imputer jobids), caps retries at 3,
# and prints a status table.
#
# Trigger via /loop:
#   /loop 1h bash /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/babysit.sh

set -euo pipefail
exec python3 "$(dirname "$0")/_babysit.py" "$@"
