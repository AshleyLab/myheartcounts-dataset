#!/bin/bash
# DLinear 7-day imputation eval. Reuses ``DLinearImputer`` (same class as
# daily DLinear); difference is ``data.n_days=7`` + a 7-day-trained
# checkpoint (W&B ``mhc-pypots-dlinear/dlinear:v48``).
#
# Note: the release directory is at ``${RELEASES}/dlinear-7day`` (kebab,
# from an earlier setup pass), while the Hydra method key is
# ``dlinear_weekly`` (snake — matches the ``lsm2_weekly_sparse`` naming
# convention). Decoupled on purpose; no code derives one from the other.

#SBATCH --job-name=mhc-impute-dlinear-weekly
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
# 7-day windows are 7× the daily memory footprint. History:
#   - 64G (daily preset): OOMed at MaxRSS=64.0G in jid 27551695
#   - 96G: OOMed at MaxRSS=96.0G in jid 27590952 (32 min in)
# 96G bound is the in-memory metric + bootstrap accumulator (10080-step
# windows × full test cohort). Two mitigations applied below:
#   - data.batch_size 2000 -> 500 and num_workers 8 -> 4 (dataloader prefetch)
#   - evaluation.compute_metrics=false (mirror MHC-benchmark's pairs-only
#     mode at jobs/.../run_imputation_dlinear_7day_max91d.sbatch:49).
#     Metrics + bootstrap are recomputed by run_paper_pipeline.py from the
#     Parquet pairs in Phase 2 — no information lost.
# Mem bumped to 128G for extra headroom on the dataloader buffers.
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=/scratch/users/schuetzn/logs/openmhc/dlinear_weekly_%j.out
#SBATCH --error=/scratch/users/schuetzn/logs/openmhc/dlinear_weekly_%j.err

# shellcheck disable=SC1091
source /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/_common.sh

nvidia-smi || true

mhc-impute-eval \
  method=dlinear_weekly \
  hydra.run.dir="${RUNS_ROOT}/dlinear_weekly" \
  output.results_dir="${RUNS_ROOT}/dlinear_weekly" \
  method.release_dir="${RELEASES}/dlinear-7day" \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  data.split_file="${SPLIT_FILE}" \
  +masking.masks_file="${MASKS}" \
  data.n_days=7 \
  data.batch_size=500 \
  data.num_workers=4 \
  data.num_eval_workers=1 \
  method.inference_batch_size=32 \
  evaluation.save_pairs=true \
  evaluation.compute_metrics=false \
  bootstrap=off \
  seed=42
