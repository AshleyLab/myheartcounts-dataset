#!/usr/bin/env bash
#SBATCH --job-name=fc-paper-pipeline
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --exclude=simurgh2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_eval/paper_pipeline_%j.out
#SBATCH --error=logs/forecasting_eval/paper_pipeline_%j.err

# Aggregation stage of the forecasting paper pipeline: discover + skill + rank
# for MHC_FORECAST_RUN_LABEL (CPU-only; reads the per-model metric parquets the
# eval jobs already wrote). Submitted by submit_pipeline.sh with an afterok
# dependency on all eval jobs. Mirrors jobs/sherlock/imputation_eval/run_paper_bootstrap.sbatch.
#
# Re-run standalone any time the eval metrics exist:
#   sbatch --export=ALL,MHC_FORECAST_RUN_LABEL=<label> run_paper_pipeline.sbatch
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/sc-cluster/forecasting_eval/_common.sh"

python scripts/paper_results/forecasting/run_paper_pipeline.py \
    --sweep-config configs/paper/sweep_forecasting.yaml \
    --skip-eval \
    --run-label "${RUN_LABEL}"

ls -la "${RUNS_ROOT}/summary/${RUN_LABEL}"
