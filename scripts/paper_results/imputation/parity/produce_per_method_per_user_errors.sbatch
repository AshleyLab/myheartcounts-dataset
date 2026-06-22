#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=produce_pu
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15
#SBATCH -o /scratch/users/schuetzn/tmp/claude/produce_pu_%a.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/produce_pu_%a.err

# Fan out the per-method per_user_errors producer across all 16 methods in
# /scratch/users/schuetzn/openmhc-imputation-eval/paper/bootstrap_method_dirs.json.
# Each array task handles one method (~10-15 min); parallel SLURM slots
# collapse the total wall time to ~one task's worth.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

# Method order MUST match paper/bootstrap_method_dirs.json. Pinned here so
# parity is reproducible without parsing JSON in bash.
METHODS=(
  mean mode locf linear temporal_mean temporal_mode
  personalized_mean personalized_mode personalized_temporal_mean
  brits dlinear dlinear_weekly fedformer timesnet
  lsm2 lsm2_weekly_sparse
)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "[array=$SLURM_ARRAY_TASK_ID] method=$METHOD"

python scripts/paper_results/imputation/parity/produce_per_method_per_user_errors.py \
    --method "$METHOD"
