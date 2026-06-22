#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=fc_parity
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_parity_%j.out
#SBATCH --error=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_parity_%j.err

# Full forecasting substrate parity gate (Gate B + Gate C-pre): the substrate-driven
# reducers must reproduce the committed forecasting_bca_20260618 CSVs (skill + rank +
# bootstraps + fairness, the latter still under the current max-min disparity).
# CPU-only, single-process pandas; ~30-45 min. Submit:
#   mkdir -p /simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs
#   sbatch scripts/paper_results/forecasting/parity/parity_substrate.sbatch

set -euo pipefail

export MHC_DATA_DIR=/simurgh/u/schuetzn/OpenMHC-Full/data
cd /simurgh/u/schuetzn/myheartcounts-dataset

/simurgh/u/schuetzn/conda/envs/openmhc/bin/python \
    scripts/paper_results/forecasting/parity/parity_substrate.py "$@"
