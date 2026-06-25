#!/bin/bash -l
#SBATCH --job-name=ds_parity
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=scratch/parity_logs/ds_parity_%j.out
#SBATCH --error=scratch/parity_logs/ds_parity_%j.err

# Downstream substrate parity gate (Gate B + Gate C): the per-user-pairs substrate must
# be byte-equal to the predictions the paper bootstrap consumes (Gate B), and a fresh
# substrate-driven 1000-draw bootstrap must reproduce the committed
# results/paper/bootstrap_draws.parquet within float32 tolerance (Gate C). CPU-only,
# single-process pandas; Gate C runs the full bootstrap (~minutes). Submit:
#   mkdir -p scratch/parity_logs
#   sbatch scripts/paper_results/downstream/parity/parity_substrate.sbatch          # both gates
#   sbatch scripts/paper_results/downstream/parity/parity_substrate.sbatch --skip-bootstrap  # Gate B only

set -euo pipefail

export MHC_DATA_DIR="${MHC_DATA_DIR:?set MHC_DATA_DIR to the dataset root}"
cd "$(git rev-parse --show-toplevel)"

python scripts/paper_results/downstream/parity/parity_substrate.py "$@"
