#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=fc_gate_d
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_gate_d_%j.out
#SBATCH --error=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_gate_d_%j.err

# Full end-to-end Gate D: run openmhc.evaluate_forecasting with a Seasonal-Naive
# forecaster on the full dataset; assert self-skill ~0 + substrate == shipped
# baseline. CPU-only, sequential evaluator. Submit:
#   mkdir -p /simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs
#   sbatch scripts/paper_results/forecasting/parity/gate_d_public_api.sbatch

set -euo pipefail

export MHC_DATA_DIR=/simurgh/u/schuetzn/OpenMHC-Full/data
cd /simurgh/u/schuetzn/myheartcounts-dataset

/simurgh/u/schuetzn/conda/envs/openmhc/bin/python \
    scripts/paper_results/forecasting/parity/gate_d_public_api.py "$@"
