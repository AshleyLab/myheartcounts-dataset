#!/bin/bash
# Cross-imputer paper-ready bootstrap (Phase 1 + Phase 2).
# Depends on every imputer sbatch via afterok in submit_all.sh.
#
# Reads pairs/ dirs from ${RUNS_ROOT}/<method>/pairs (written by the per-
# method sbatches), produces:
#   ${PAPER_OUT}/bootstrap_draws.parquet
#   ${PAPER_OUT}/avg_rankings_bootstrap.csv
#   ${PAPER_OUT}/skill_scores_bootstrap.csv
#   ${PAPER_OUT}/fairness_skill_score_bootstrap.csv  # disparity-ratio (leaderboard)
#
# Re-running with --skip-eval --skip-phase1 re-aggregates from the existing
# bootstrap_draws.parquet (cheap), or with --skip-eval alone re-bootstraps
# (n_boot * methods * scenarios) from the existing pairs/ — no re-eval.

#SBATCH --job-name=mhc-impute-paper-bootstrap
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/scratch/users/%u/logs/openmhc/paper_bootstrap_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/paper_bootstrap_%j.err

# shellcheck disable=SC1091
# Use absolute path: sbatch copies the script to /var/spool/slurmd at run
# time, so $BASH_SOURCE doesn't point at the in-repo sibling _common.sh.
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

python scripts/paper_results/imputation/run_paper_pipeline.py \
  --sweep-config jobs/sherlock/imputation_eval/sweep_methods.yaml \
  --skip-eval

ls -la "${PAPER_OUT}"
