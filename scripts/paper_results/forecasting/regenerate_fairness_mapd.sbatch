#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --job-name=fc_fair_mapd
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_fair_mapd_%j.out
#SBATCH --error=/simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs/fc_fair_mapd_%j.err

# Regenerate the canonical forecasting fairness CSVs under the MAPD disparity.
# Skill/rank are unaffected (proven by the substrate parity gate); only the two
# fairness CSVs are replaced (old max-min backed up to *.maxmin.bak). Submit:
#   mkdir -p /simurgh/u/schuetzn/myheartcounts-dataset/scratch/parity_logs
#   sbatch scripts/paper_results/forecasting/regenerate_fairness_mapd.sbatch

set -euo pipefail

export MHC_DATA_DIR=/simurgh/u/schuetzn/OpenMHC-Full/data
cd /simurgh/u/schuetzn/myheartcounts-dataset

/simurgh/u/schuetzn/conda/envs/openmhc/bin/python \
    scripts/paper_results/forecasting/regenerate_fairness_mapd.py "$@"
