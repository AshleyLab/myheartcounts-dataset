# Sherlock forecasting-model training

Sherlock counterpart of `jobs/simurgh/forecasting_train/`. Trains the three
trainable PyPOTS forecasters (**DLinear, MixLinear, SegRNN**) and emits release
bundles the eval pipeline consumes.

Like the Simurgh jobs, training uses `training.include_short_history=true` (the
default) so the training input distribution matches what the evaluator feeds
(short histories NaN-left-padded to `n_steps`) — the basis for a fair comparison
against the foundation models.

## Train

```bash
sbatch jobs/sherlock/forecasting_train/run_dlinear.sbatch
sbatch jobs/sherlock/forecasting_train/run_mixlinear.sbatch
sbatch jobs/sherlock/forecasting_train/run_segrnn.sbatch
```

Bundles land under `/scratch/users/$USER/openmhc-forecasting-train/releases/<model>_<timestamp>/`
(`model.pypots` + `standard_scaler_stats.json` + `training_config.json` +
`openmhc_manifest.json`). The release dir is printed as `RELEASE_DIR=<path>`.

## Swap into eval

```bash
export MHC_FORECAST_DLINEAR_RELEASE_DIR=/scratch/users/$USER/openmhc-forecasting-train/releases/dlinear_<timestamp>
sbatch jobs/sherlock/forecasting_eval/run_dlinear.sbatch
```

## Notes

- Resource budget mirrors the imputation-train jobs: `gpu` partition, Tesla-only
  (`GPU_BRD:TESLA`) to dodge consumer-RTX requeues. These linear/RNN forecasters
  are small; 24h is generous.
- **MixLinear** has no `optimizer` kwarg in PyPOTS (`training.optimizer_lr` not
  applied; uses PyPOTS' default optimizer).
- W&B on by default; disable with `MHC_FORECAST_WANDB=false`.
