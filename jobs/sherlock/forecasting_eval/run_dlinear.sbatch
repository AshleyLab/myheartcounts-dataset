#!/usr/bin/env bash
#SBATCH --job-name=forecast-dlinear
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/forecasting_eval/dlinear_%j.out
#SBATCH --error=logs/forecasting_eval/dlinear_%j.err

set -euo pipefail

export MHC_LOAD_CUDA=1
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

nvidia-smi || true
override="$(require_release_override dlinear)"
run_forecast_model dlinear "${override}" "$@"
