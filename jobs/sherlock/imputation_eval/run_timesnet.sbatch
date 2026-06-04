#!/bin/bash
# TimesNet imputation eval (PyPOTS Transformer with FFT, ~4-8 GB VRAM).
# Request a 24 GB+ GPU for headroom on the FFT path.

#SBATCH --job-name=mhc-impute-timesnet
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
# Avoid consumer-grade GPUs (RTX 3090 / Titan Xp). The Tesla cards
# (P100/V100/A100/L40S/H100) almost never throw "GPU is lost" NVML errors
# that cause silent SLURM requeues mid-inference.
# Note: SLURM matches GPU_MEM:<tier> by exact string, not as ">=". On
# Sherlock the Tesla-family tiers >=32GB are {32GB,48GB,80GB} — enumerate
# them via OR to maximize the eligible pool.
#SBATCH --constraint=GPU_BRD:TESLA&(GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB)
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/timesnet_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/timesnet_%j.err

# shellcheck disable=SC1091
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

nvidia-smi || true

mhc-impute-eval \
  method=timesnet \
  hydra.run.dir="${RUNS_ROOT}/timesnet" \
  output.results_dir="${RUNS_ROOT}/timesnet" \
  method.release_dir="${RELEASES}/timesnet" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.batch_size=2000 \
  data.num_workers=8 \
  data.num_eval_workers=1 \
  method.inference_batch_size=64 \
  evaluation.save_pairs=true \
  bootstrap=on \
  seed=42
