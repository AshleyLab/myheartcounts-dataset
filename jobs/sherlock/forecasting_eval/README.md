# Sherlock Forecasting Evaluation

This directory contains SLURM wrappers for the public Hydra forecasting CLI,
`mhc-forecast-eval`.

## Files

- `_common.sh`: shared environment setup, repo/data paths, and output roots.
- `run_baselines.sbatch`: CPU job for `seasonal_naive`, `autoARIMA`, and
  `autoETS`.
- `run_chronos2.sbatch`, `run_toto.sbatch`, `run_mixlinear.sbatch`,
  `run_dlinear.sbatch`, `run_segrnn.sbatch`: one GPU job per model family.
- `aggregate_results.sbatch`: summary aggregation over completed metric outputs.
- `submit_all.sh`: submits all model jobs and, by default, chains aggregation
  with an `afterok` dependency.

## Usage

Submit the full suite:

```bash
jobs/sherlock/forecasting_eval/submit_all.sh
```

Submit baselines only:

```bash
sbatch jobs/sherlock/forecasting_eval/run_baselines.sbatch
```

Use a fixed run label:

```bash
MHC_FORECAST_RUN_LABEL=paper_retry_001 \
jobs/sherlock/forecasting_eval/submit_all.sh
```

Disable automatic aggregation:

```bash
MHC_FORECAST_AGGREGATE=0 \
jobs/sherlock/forecasting_eval/submit_all.sh
```

## Environment

Important variables:

| Variable | Meaning |
|---|---|
| `MHC_REPO_DIR` | Repo checkout. Defaults to the script-resolved repo root. |
| `MHC_VENV` | Python virtualenv. Defaults to `/scratch/users/$USER/envs/mhc-benchmark` if present. |
| `MHC_DATA_DIR` | Full data cache containing `hourly_trajectory`, `splits`, and `forecasting_sample_index`. Defaults to `~/.cache/openmhc/data-full` when available. |
| `MHC_FORECAST_RUNS_ROOT` | Forecasting output root. Defaults to `results/forecasting_eval/sherlock`. |
| `MHC_FORECAST_RUN_LABEL` | Shared run label used by all submitted jobs. |
| `MHC_FORECAST_AGGREGATE` | Set to `0` to skip automatic aggregation. |

Learned model checkpoint releases can be supplied with:

```bash
export MHC_FORECAST_DLINEAR_RELEASE_DIR=/path/to/openmhc-dlinear-forecast
export MHC_FORECAST_MIXLINEAR_RELEASE_DIR=/path/to/openmhc-mixlinear-forecast
export MHC_FORECAST_SEGRNN_RELEASE_DIR=/path/to/openmhc-segrnn-forecast
```

Foundation-model release overrides use the same pattern:

```bash
export MHC_FORECAST_CHRONOS2_RELEASE_DIR=/path/to/openmhc-chronos2-forecast
export MHC_FORECAST_TOTO_RELEASE_DIR=/path/to/openmhc-toto-forecast
```

## Metrics Modes

The main skill and fairness summaries use per-task metrics by default: phone/watch
step count and distance are scored as separate channels and combined into
`steps`/`distance` scopes by geometric mean (consistent with the imputation track).

Pass `--combine-channels` for the legacy signal-merge mode (phone/watch nan-meaned
before scoring), used for some appendix hour-group tables:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths seasonal_naive_0=/path/to/predictions/seasonal_naive \
  --metrics-output-path results/metrics_combined \
  --combine-channels
```
