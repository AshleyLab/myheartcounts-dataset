# Simurgh (SC) Forecasting Evaluation — CPU baselines

SLURM wrappers for the public Hydra forecasting CLI (`mhc-forecast-eval`),
adapted from `jobs/sherlock/forecasting_eval/` for the Simurgh cluster.

Covers the four **CPU-only** Track-3 baselines:
`seasonal_naive`, `seasonal_naive_average_history`, `autoARIMA`, `autoETS`.
GPU models (chronos2/toto/mixlinear/dlinear/segrnn) are out of scope here.

## Files

- `_common.sh` — conda activation, dataset root, BLAS pinning, output roots, and
  the `run_forecast_model` helper.
- `run_naive.sbatch` — `seasonal_naive` + `seasonal_naive_average_history`
  (2 cpus, 32G, 4h).
- `run_autoets.sbatch` — `autoETS` (8 cpus, 64G, 24h).
- `run_autoarima.sbatch` — `autoARIMA` (16 cpus, 64G, 48h; the long pole,
  resumable).
- `aggregate_results.sbatch` — cross-model summary over completed metrics.
- `submit_all.sh` — submits the 3 CPU jobs + chains aggregation (`afterok`).

## Prerequisites (one-time)

The `openmhc` conda env needs the statistical-model deps:

```bash
/simurgh/u/schuetzn/conda/envs/openmhc/bin/pip install sktime pmdarima
```

## Usage

Full CPU suite (recommended):

```bash
jobs/simurgh/forecasting_eval/submit_all.sh
```

A single model:

```bash
sbatch jobs/simurgh/forecasting_eval/run_autoarima.sbatch
```

Fixed run label / skip aggregation:

```bash
MHC_FORECAST_RUN_LABEL=paper_001 jobs/simurgh/forecasting_eval/submit_all.sh
MHC_FORECAST_AGGREGATE=0          jobs/simurgh/forecasting_eval/submit_all.sh
```

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `MHC_REPO_DIR` | Repo checkout | script-resolved repo root |
| `MHC_CONDA_BASE` | conda install prefix | `/simurgh/u/schuetzn/conda` |
| `MHC_CONDA_ENV` | conda env name | `openmhc` |
| `MHC_DATA_DIR` | dataset root (`hourly_trajectory`, `splits`, `forecasting_sample_index`, `labels`) | `/simurgh/u/schuetzn/OpenMHC-Full/data` |
| `MHC_FORECAST_RUNS_ROOT` | output root | `results/forecasting_eval/simurgh` |
| `MHC_FORECAST_RUN_LABEL` | shared run label | `forecasting_<jobid>` |
| `MHC_FORECAST_OVERWRITE` | overwrite existing parquets (else resume by skipping) | `false` |
| `MHC_FORECAST_AGGREGATE` | set `0` to skip aggregation | `1` |

## Notes

- Account/partition: `--account=simurgh --partition=simurgh` (CPU-only — no
  `--gres` requested). Core counts are kept modest: the harness loop is
  sequential, so only `autoARIMA` (and marginally `autoETS`) benefit from cores.
- BLAS threads are pinned to 1 in `_common.sh` so joblib (`n_jobs=-1`) does not
  oversubscribe.
- Each run persists raw `(ground_truth, prediction)` pairs plus a `scales.npz`
  under its `*_metrics/<RUN_LABEL>/` dir, enabling post-hoc metric recomputation
  and bootstrapping without re-running models.
- `autoARIMA` is resumable: re-submitting continues from where it stopped
  (existing per-user parquets are skipped while `MHC_FORECAST_OVERWRITE=false`).
