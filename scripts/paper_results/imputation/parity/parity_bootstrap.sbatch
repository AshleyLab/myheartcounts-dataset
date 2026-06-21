#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=parity_bootstrap
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/schuetzn/tmp/claude/parity_bootstrap.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/parity_bootstrap.err

# §3 Bootstrap parity. Re-runs bootstrap_imputation_draws.py via the Phase B
# fast path (--per-user-errors-dir) into paper-verification/bootstrap/ and
# byte-diffs against paper/bootstrap_draws.parquet.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

PAPER=/scratch/users/schuetzn/openmhc-imputation-eval/paper
VERIF=/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification
mkdir -p "$VERIF/bootstrap"

echo "=== (1/2) Phase B bootstrap → $VERIF/bootstrap/bootstrap_draws.parquet ==="
python scripts/paper_results/imputation/bootstrap_imputation_draws.py \
    --method-dirs "$PAPER/bootstrap_method_dirs.json" \
    --per-user-errors-dir "$VERIF/per_user/" \
    --output "$VERIF/bootstrap/bootstrap_draws.parquet" \
    --per-user-errors "$VERIF/bootstrap/per_user_errors.parquet" \
    --n-boot 1000 --seed 42 --splits test \
    --age-bins 18 30 40 50 60 \
    --strict

echo "=== (2/2) Parity diff ==="
python scripts/paper_results/imputation/parity/parity_bootstrap.py \
    --new "$VERIF/bootstrap/bootstrap_draws.parquet" \
    --gt  "$PAPER/bootstrap_draws.parquet"
