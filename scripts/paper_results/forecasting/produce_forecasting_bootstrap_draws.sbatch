#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=fc_boot_draws
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_boot_draws_%j.out
#SBATCH --error=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_boot_draws_%j.err

# Build the forecasting bootstrap-draws reference (forecasting/bootstrap/draws.parquet).
# CPU-only; ~30 min. Submit:
#   sbatch scripts/paper_results/forecasting/produce_forecasting_bootstrap_draws.sbatch

set -euo pipefail

export MHC_DATA_DIR=/simurgh/u/schuetzn/OpenMHC-Full/data
cd /simurgh/u/schuetzn/myheartcounts-dataset

/simurgh/u/schuetzn/conda/envs/openmhc/bin/python \
    scripts/paper_results/forecasting/produce_forecasting_bootstrap_draws.py "$@"
