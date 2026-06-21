#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=parity_locf
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/users/schuetzn/tmp/claude/parity_locf.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/parity_locf.err

# §1 LOCF per-user-errors parity gate. See parity_locf.py module docstring.
#
# Per ~/.claude/CLAUDE.md: ml purge first (the login shell auto-loads
# py-3.9 modules whose site-packages poison Py-3.12 sys.path); unset
# PYTHONPATH; load python/3.12.1 so libpython3.12.so.1.0 resolves; then
# activate the project venv.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate

# labels.api needs <dataset_root>/labels/enrollment_info.json to bucket
# users by age_group. The existing paper run used the full dataset; the
# locf manifest references the full-dataset user_ids, so the lookup root
# must match (xs would silently fall through to "unknown" buckets and
# wreck parity).
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

python scripts/paper_results/imputation/parity/parity_locf.py "$@"
