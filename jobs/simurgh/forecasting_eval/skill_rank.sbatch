#!/usr/bin/env bash
#SBATCH --job-name=fc-skill-rank
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --exclude=simurgh2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/forecasting_eval/skill_rank_%j.out
#SBATCH --error=logs/forecasting_eval/skill_rank_%j.err

# Canonical Layer-2 entry point. From ONE RUN_LABEL's metrics tree, compute the
# benchmark skill score (continuous=mae + binary=auprc, vs seasonal_naive) AND
# the grouped mean-rank, in a single job. Both read the per-model
# `<model>_metrics/<LABEL>/{mae,auprc,...}` tree that `mhc-forecast-eval` now
# emits (point + binary co-located; see config.MetricsConfig).
#
# Assumes every model — including the baseline — was evaluated under the same
# RUN_LABEL. Override knobs via env: MHC_FORECAST_BASELINE, MHC_FORECAST_CONT_METRICS,
# MHC_FORECAST_BIN_METRICS.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/simurgh/forecasting_eval/_common.sh"

BASELINE="${MHC_FORECAST_BASELINE:-seasonal_naive}"
CONT_METRICS="${MHC_FORECAST_CONT_METRICS:-mae}"
BIN_METRICS="${MHC_FORECAST_BIN_METRICS:-auprc}"
summary_dir="${RUNS_ROOT}/summary/${RUN_LABEL}"
models_json="${summary_dir}/skill_rank_models.json"
mkdir -p "${summary_dir}"

# Discover model -> metrics dir for this RUN_LABEL and write a models.json.
python - "${RUNS_ROOT}" "${RUN_LABEL}" "${models_json}" <<'PY'
import json, sys
from pathlib import Path

runs_root, label, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
models: dict[str, str] = {}
for model_dir in sorted(runs_root.iterdir()):
    if not model_dir.is_dir() or model_dir.name == "summary":
        continue
    name = model_dir.name
    metrics_dir = model_dir / "predictions" / label / f"{name}_metrics" / label
    if (metrics_dir / "mae").is_dir():
        models[name] = str(metrics_dir)
if not models:
    raise SystemExit(f"No model metrics found for RUN_LABEL={label} under {runs_root}")
Path(out).write_text(json.dumps({"models": models}, indent=2))
print(f"discovered {len(models)} models: {', '.join(sorted(models))}")
PY

echo "=== Skill score (continuous=${CONT_METRICS}, binary=${BIN_METRICS}, baseline=${BASELINE}) ==="
python src/forecasting_evaluation/metrics/skill_score_summary.py \
    --config "${models_json}" \
    --baseline "${BASELINE}" \
    --continuous-metrics ${CONT_METRICS} \
    --binary-metrics ${BIN_METRICS} \
    --aggregation-unit user \
    --output-dir "${summary_dir}" \
    --output-prefix forecasting_skill_score

echo "=== Grouped metric rank (continuous=${CONT_METRICS}, binary=${BIN_METRICS}) ==="
python src/forecasting_evaluation/metrics/grouped_metric_rank_summary.py \
    --config "${models_json}" \
    --continuous-metrics ${CONT_METRICS} \
    --binary-metrics ${BIN_METRICS} \
    --output-dir "${summary_dir}" \
    --output-prefix forecasting_grouped_metric_rank

echo "=== Outputs under ${summary_dir} ==="
ls -la "${summary_dir}"
