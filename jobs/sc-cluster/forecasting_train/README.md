# Simurgh forecasting-model training

Train the three trainable PyPOTS forecasters (**DLinear, MixLinear, SegRNN**) on
the OpenMHC dataset and emit release bundles the eval pipeline consumes directly.

Unlike the published checkpoints, these models are trained with
`training.include_short_history=true` (the default): windows whose history is
shorter than `n_steps` are **NaN-left-padded** rather than dropped, so the
training input distribution matches what the evaluator feeds at inference time
(`BasePyPOTSForecastingModel.predict`). This is what makes the benchmark
comparison against the foundation models (Toto/Chronos, which see all windows)
fair.

## Layout

```
jobs/sc-cluster/forecasting_train/
├── _common.sh            # sourced env (conda + paths + run_forecast_train helper)
├── run_dlinear.sbatch
├── run_mixlinear.sbatch
├── run_segrnn.sbatch
└── README.md             # this file
```

## Train

```bash
sbatch jobs/sc-cluster/forecasting_train/run_dlinear.sbatch
sbatch jobs/sc-cluster/forecasting_train/run_mixlinear.sbatch
sbatch jobs/sc-cluster/forecasting_train/run_segrnn.sbatch
```

Each job writes a timestamped release bundle under
`results/forecasting_train/simurgh/releases/<model>_<timestamp>/` containing:

```
<model>_<timestamp>/
├── model.pypots                # trained checkpoint
├── standard_scaler_stats.json  # train-fit channel StandardScaler
├── training_config.json        # arch contract the eval adapter reads
└── openmhc_manifest.json       # spec_version=1 (openmhc.forecasters._release)
```

The release dir is printed in the log as `RELEASE_DIR=<path>`.

## Swap into the eval pipeline

The eval jobs read a per-model release dir from
`MHC_FORECAST_<MODEL>_RELEASE_DIR`:

```bash
export MHC_FORECAST_DLINEAR_RELEASE_DIR=results/forecasting_train/simurgh/releases/dlinear_<timestamp>
sbatch jobs/sc-cluster/forecasting_eval/run_dlinear.sbatch
# likewise MHC_FORECAST_MIXLINEAR_RELEASE_DIR / MHC_FORECAST_SEGRNN_RELEASE_DIR
```

## Smoke test (before committing GPU time)

```bash
source jobs/sc-cluster/forecasting_train/_common.sh
mhc-forecast-train model=dlinear \
  training.epochs=1 training.batch_size=8 data.max_samples=200 \
  output.saving_path=/tmp/ft_smoke output.release_dir=/tmp/ft_smoke_release \
  output.wandb_enabled=false
mhc-forecast-eval model=dlinear model.release_dir=/tmp/ft_smoke_release data.max_samples=200
```

## Notes

- **MixLinear** has no `optimizer` kwarg in PyPOTS, so `training.optimizer_lr`
  does not reach it — it trains with PyPOTS' internal default optimizer. DLinear
  and SegRNN honor `optimizer_lr`.
- W&B is on by default (`output.wandb_enabled=true`); PyPOTS' TensorBoard scalars
  stream into the run via `sync_tensorboard`. Disable with
  `MHC_FORECAST_WANDB=false` (or `output.wandb_enabled=false`).
- Train and eval read the same `MHC_DATA_DIR`, `split_file`, and
  `sample_index_file`, so splits are identical by construction.
