#!/bin/bash
# 16-method (no lsm2_weekly dense) cross-imputer paper-ready bootstrap.
# Sibling of run_paper_bootstrap.sbatch. Outputs to paper_no_dense/ so the
# canonical 17-method paper/ isn't touched.

#SBATCH --job-name=mhc-impute-paper-bootstrap-no-dense
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/scratch/users/%u/logs/openmhc/paper_bootstrap_no_dense_%j.out
#SBATCH --error=/scratch/users/%u/logs/openmhc/paper_bootstrap_no_dense_%j.err

# Use absolute path: sbatch copies the script to /var/spool/slurmd at run
# time, so $BASH_SOURCE doesn't point at the in-repo sibling _common.sh.
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

python scripts/paper_results/imputation/run_paper_pipeline.py \
  --sweep-config jobs/sherlock/imputation_eval/sweep_methods_no_dense.yaml \
  --skip-eval

ls -la "${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper_no_dense"
