#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=parity_point
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/schuetzn/tmp/claude/parity_point.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/parity_point.err

# §2 Point-metrics parity. Runs compute_imputation_paper_metrics.py twice:
#  (i) the new B3 path with --per-user-errors against the fan-out files at
#      paper-verification/per_user/
#  (ii) the legacy from-pairs path using paper/bootstrap_method_dirs.json
#       (READ-ONLY input)
# Then diffs the resulting CSVs against each other (self-consistency, A) and
# against paper/*_bootstrap.csv mean ± 5*se (statistical, B). See
# parity_point.py docstring for the full contract.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

PAPER=/scratch/users/schuetzn/openmhc-imputation-eval/paper
VERIF=/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification
mkdir -p "$VERIF/point_pue" "$VERIF/point_pairs"

echo "=== (1/3) B3 path: --per-user-errors → $VERIF/point_pue/ ==="
python scripts/paper_results/compute_imputation_paper_metrics.py \
    --per-user-errors "$VERIF/per_user/" \
    --output-dir "$VERIF/point_pue/" \
    --splits test \
    --strict

echo "=== (2/3) From-pairs path: --method-dirs → $VERIF/point_pairs/ ==="
python scripts/paper_results/compute_imputation_paper_metrics.py \
    --method-dirs "$PAPER/bootstrap_method_dirs.json" \
    --output-dir "$VERIF/point_pairs/" \
    --splits test \
    --strict

echo "=== (3/3) Parity diff ==="
python scripts/paper_results/imputation/parity/parity_point.py \
    --new "$VERIF/point_pue" \
    --pairs "$VERIF/point_pairs" \
    --bootstrap "$PAPER"
