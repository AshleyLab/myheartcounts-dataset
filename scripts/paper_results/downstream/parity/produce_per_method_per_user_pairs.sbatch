#!/bin/bash -l
#SBATCH --job-name=ds_produce_pairs
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=scratch/parity_logs/ds_produce_pairs_%j.out
#SBATCH --error=scratch/parity_logs/ds_produce_pairs_%j.err

# Build the 8 downstream leaderboard substrate parquets (+ provenance sidecars) from
# the saved eval predictions. CPU-only, single process (pooling parquets is light) —
# all 8 methods in one task. Submit:
#   mkdir -p scratch/parity_logs
#   sbatch scripts/paper_results/downstream/parity/produce_per_method_per_user_pairs.sbatch

set -euo pipefail

# Dataset root (predictions are repo-relative, so this is only needed if you override
# --predictions-dir to a path that resolves against the dataset).
export MHC_DATA_DIR="${MHC_DATA_DIR:?set MHC_DATA_DIR to the dataset root}"
cd "$(git rev-parse --show-toplevel)"

python scripts/paper_results/downstream/parity/produce_per_method_per_user_pairs.py "$@"
