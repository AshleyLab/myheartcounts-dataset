# Forecasting Evaluation — Internals

> Deep-dive companion to [README.md](README.md). The README covers how to *run*
> evaluation (Hydra CLI, config schema, checkpoints, quick start); this document
> covers how the pipeline *works* internally: the preprocessing chain, a
> code-linked walkthrough of `ForecastingEvaluator.run()`, and the offline metric
> math. Code line references may drift over time — treat them as starting points.

## 1. Data Preprocessing

### 1.1 Raw Inputs

Evaluation depends on two HuggingFace datasets that are already materialized on disk:

- `data/processed/daily_hf`: day-level data, used to build the day-level retain mask.
- `data/hourly_trajectory`: user-organized hourly trajectories, used for evaluation, window slicing, and metric ground truth.

Each record in `hourly_trajectory` must include at least:

- `user_id`
- `values`, with shape `(T, C)`
- `timestamps`
- `channel_names`

### 2.1.1 How hourly trajectory is built

`data/hourly_trajectory` is not read directly from raw files. It is produced by [scripts/data/build_hourly_trajectory_res.py](../../scripts/data/build_hourly_trajectory_res.py#L65), which calls [build_hourly_trajectory_res()](../data/processing/hourly_trajectory_res.py#L19). The Imperial PBS entrypoint is [build_hourly_trajectory.pbs](../../jobs/imperial/pbs/forecasting/build_hourly_trajectory.pbs#L1).

The full processing chain is:

1. Raw HDF5 -> `data/processed/daily_hf`:
2. `data/processed/daily_hf` -> `data/hourly_trajectory`: then [build_hourly_trajectory()](../data/processing/hourly_hf_trajectory.py#L116) is called. Data is sorted by `user_id, date`, then concatenated user by user in a streaming way. Each user's daily `(19, 1440)` minute-level values are aggregated by [resample_day_to_hourly()](../data/processing/daily_hf_to_daily_hourly_hf.py#L90) into `(19, 24)`:
   - Binary channels (sleep/workout), i.e., channels `7-18`, use hourly `nanmax`: if any minute in an hour is 1, the hour is 1; if the whole hour is missing, it is `NaN`.
   - Accumulation channels (for example steps, distance, flights, active energy) use hourly `nansum`; if the whole hour is missing, it is `NaN`.
   - Rate channels (heart rate) use hourly `nanmean`.
Existing calendar dates are inserted as real aggregated 24-hour blocks; missing calendar days in between are filled with all-`NaN` blocks of shape `(24, 19)` to keep a continuous timeline.
3. Finally, one trajectory record is output per user:
   - `values`: `(T, 19)`, where `T = number_of_calendar_days_from_first_to_last * 24`
   - `timestamps`: ISO hourly timestamps of length `T`
   - `channel_names`: canonical 19-channel names
   - `user_id`

Important: this hourly trajectory build path itself does not perform forecasting sample filtering and does not create train/val/test splits. It only organizes raw minute-level daily data into continuous user-level hourly sequences. Which days can be used as forecasting start points is decided later by the day retain mask and sample index.

Default paths are defined in [configs/forecasting_eval/default.yaml](../../configs/forecasting_eval/default.yaml#L1) and [config.py](config.py#L13).

### 2.2 Day retain mask

[generate_day_drop_mask()](data/day_retain_mask_generator.py#L19) is built from daily HF data and outputs:

```text
data/forecasting_sample_index/day_remain_mask.json
```

Format:

```json
{
  "user_id": ["YYYY-MM-DD", "..."]
}
```

It represents which days for each user pass day-level quality control. Default day-level filtering in code includes:

- `WearTimeFilter(min_wear_fraction=0.5)`
- `LowChannelVarianceFilter`

Note: [scripts/precompute_forecasting_inputs.py](../../scripts/precompute_forecasting_inputs.py#L105) currently parses arguments like `--min_wear_fraction` and `--disable_variance_filter`, but does not pass them through when calling [generate_day_drop_mask()](data/day_retain_mask_generator.py#L19), so function defaults are still used in practice.

### 2.3 Sample index

[SampleIndexGenerator](data/sample_index_generator.py#L24) reads hourly trajectory and day retain mask, and produces:

```text
data/forecasting_sample_index/sample_index_P_24_M_H_7_3_S_100.json
```

Format:

```json
{
  "user_id": [1, 2, 3]
}
```

Each integer is a forecast-start day index. Generation logic:

1. Raw candidates: for each user, generate `1..max_start_day` first.
   `max_start_day = (total_hours - forecasting_length) // 24`, so a full prediction horizon must remain from that day boundary.
2. `missing_mask` filter (the `_M` in filename): all future calendar days covered by the prediction window must appear in `day_remain_mask`. Here, `M` is a missing-mask constraint that limits forecast-target future days to day-level QC-retained days. It is not a strong constraint requiring every channel in hourly targets to be observed; hourly missing values are still excluded later by `NaN`/observed-mask handling in metrics. If `forecasting_length=48`, the next 2 days are checked.
3. `historical_check` filter: among the most recent `recent_day_count` days before target day, at least `minimum_valid_day` days must appear in `day_remain_mask`. For example, default `H_7_3` means at least 3 valid days in the previous 7 days.
4. `maximum_sample_count_per_user` filter: keep up to the configured number of candidate days per user; if exceeded, randomly sample. Default `_S_100`.
5. Users with empty candidate lists are dropped when saving.

Imperial entrypoint example is in [generate_sample_index.slurm](../../jobs/imperial/slurm/forecasting/generate_sample_index.slurm#L1):

```bash
python scripts/precompute_forecasting_inputs.py \
  --daily_hf_dir data/processed/daily_hf \
  --hourly_trajectory_path data/hourly_trajectory \
  --sample_index_path data/forecasting_sample_index/sample_index.json \
  --day_remain_mask_path data/forecasting_sample_index/day_remain_mask.json \
  --forecasting_length 24 \
  --filter_parameters_json '{"missing_mask": true, "historical_check": {"recent_day_count": 7, "minimum_valid_day": 3}, "maximum_sample_count_per_user": 100}' \
  --seed 42
```

## 3. Walk Through the Evaluation Flow Step by Step via ForecastingEvaluator.run()

This section is organized by the execution order of [ForecastingEvaluator.run()](evaluation/evaluator.py#L51), so you can map it directly to code.

### 3.1 Step 0: Print config and create model

The entry first calls [print_config()](config.py#L327), then instantiates a model via [create_forecasting_model()](models/registry.py#L23).

Three config blocks are passed together into model construction:

- `model`
- `forecasting`
- `features`

So downstream behavior such as window length, quantile output capability, and whether fixed history windows are required (PyPOTS) is determined here.

### 3.2 Step 1: Load train/val/test together (not test-only)

`run()` enters [_load_evaluation_data()](evaluation/evaluator.py#L84), then calls [ForecastingDataLoader.load_splits()](data/data_loader.py#L59).

`load_splits()` does:

1. Validate that `data.sample_index_file` exists.
2. Read hourly trajectory from `data.trajectory_hf_dir`.
3. If `data.max_samples` is set, apply HF dataset prefix truncation first (for debugging).
4. Read sample index and keep only users appearing in the sample index.
5. Prefer loading user split from `data.split_file`; if missing, perform random split using `train_ratio / val_ratio / split_seed`.
6. Intersect each split with sample-index-eligible users.
7. Return `train_ds, val_ds, test_ds`.

`load_splits()` returns all three splits, but evaluation now builds only the
**test**-split raw history cache (the prior train/val/test standardized bundle
and its scaler-stats step are no longer part of the eval path).

### 3.4 Step 2: Build/reuse history_cf cache (test split only)

Then [prepare_history_cf_raw_cache_for_split()](data/cache_bundle.py) prepares the **raw** history cache and row-group manifest for the **test split only**, via one model-agnostic path. There is no per-model cache selection and no standardized-cache variant: every model reads the same raw full-trajectory history and the same data-quality-only window manifest (manifest schema `v2`). Models that need a fixed context window slice it themselves; models trained on standardized inputs standardize internally at predict time.

The cache still lives under:

```text
data/processed/forecasting_eval_h5/history_cf_cache/<hash>/
```

but only the raw **test** H5 and its row-group manifest are built for evaluation (no train/val or `_standard` variants).

The evaluator stores two key objects in context for later sequential inference:

- `test_cache_path`: the raw test H5 (same for every model)
- `test_row_groups`: window list grouped by trajectory row

### 3.5 Step 3: Initialize writers and enter sequential inference

`run()` then creates [PublicWriter](io/predict_result_writer.py#L120) and calls [_run_sequential()](evaluation/evaluator.py#L164).

Inside `_run_sequential()`:

1. Resolve runtime offset first, via [_resolve_runtime_daily_start_hour_offset()](evaluation/evaluator.py#L278).
2. Open H5 at `test_cache_path` (shared read).
3. Iterate over `test_row_groups` (each row group usually corresponds to one test trajectory/user).
4. For each user, create [PredictResultWriter](io/predict_result_writer.py#L220) and append multiple windows row by row into parquet.

### 3.6 Step 4: Compute real window boundaries per window and run prediction

For each `current_day` in a row group, evaluator calls [_resolve_window_hours()](data/online_dataset.py#L361):

```python
history_end_hour = current_day * 24 + daily_start_hour_offset
pred_end_hour = history_end_hour + forecasting_length
```

Data-quality validity filtering is applied (the same drops for every model — no
model-capability filtering):

- Skip if `history_end_hour <= 0`
- Skip if `pred_end_hour > trajectory_length`

Window semantics are now model-agnostic — every model receives the same
full-prefix history and owns any truncation/padding it needs:

- `history_window = history_cf[:, :history_end_hour]` (full prefix, all models)
- `target_window = history_cf[:, history_end_hour:pred_end_hour]`

The model is then invoked through the unified call path
`_invoke_forecaster(model, history_window, prediction_hours, meta)`, which
forwards only the optional metadata kwargs the model's
`predict(history, horizon, ...)` declares, times the call, tracks peak memory,
and normalizes the return to `(point, quantiles)`. Any `NaN` cells in the point
forecast are filled with the Seasonal-Naive baseline so every in-scope cell is
scored; the per-channel substitution rate is logged and surfaced as
`overall_fallback_rate` / `fallback_rate`.

When writing each prediction row, `history_length` is persisted as the absolute forecast origin `history_end_hour` (not model-consumed context length). This guarantees unambiguous reconstruction of ground-truth slices in offline metrics.

## 5. Prediction output

[PublicWriter](io/predict_result_writer.py#L120) and [PredictResultWriter](io/predict_result_writer.py#L220) write to:

```text
results/forecasting_eval/<experiment_name>/<model_name>/<user_id>.parquet
```

If `experiment_name` starts with `Test`, the model directory name appends a timestamp.

Default is `output.overwrite_existing_parquet=false`. If a user parquet already exists, that user is skipped. Set to `true` to overwrite.

Each row corresponds to one forecasting window, with major fields:

- `user_id`
- `model`
- `history_length`: absolute hour index of forecast origin in the full trajectory.
- `point_predictions`: expected shape `(n_features, forecasting_length)`.
- `quantile_predictions`: expected shape `(n_features, forecasting_length, n_quantiles)`, can be empty.
- `fallback_mask`: boolean `(n_features, forecasting_length)`; `True` where the model emitted `NaN` and the Seasonal-Naive baseline was substituted before scoring.
- `performance`: auxiliary inference metadata recorded by the harness, such as runtime and memory.

Entrypoint example:

```bash
python scripts/run_forecasting_eval.py \
  --config configs/forecasting_eval/default.yaml \
  --config configs/forecasting_eval/model_config/seasonalNaive.yaml \
  --config configs/forecasting_eval/forecasting_setup/final_eval_24.yaml
```

Imperial array entrypoint is [run_forecasting_eval.slurm](../../jobs/imperial/slurm/forecasting/run_forecasting_eval.slurm#L1), typically stacking base config, model config, and forecasting setup config.

## 6. Supported model types

See [README.md](README.md#6-supported-model-types) for the canonical `model.type`
list. If `model.name` is empty, output directory naming and the parquet `model`
field default to `model.type`.

## 7. Offline metric computation

Entrypoint:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths seasonal_naive=results/forecasting_eval/Final_Eval_24/seasonal_naive \
  --metrics-output-path results/metrics
```

`--evaluation-result-paths` uses `run_key=prediction_model_dir`. `run_key` becomes the output directory name for offline metrics.

### 7.1 Reconstructing offline pipeline inputs

For each run, [OfflineMetricsPipeline](metrics/offline/pipeline.py#L57) does:

1. Read `config.yaml` from prediction directory.
2. Re-run [ForecastingDataLoader.load_splits()](data/data_loader.py#L59) with that config.
3. Reuse the same evaluator path [prepare_history_cf_cache_bundle()](data/cache_bundle.py#L52) to load raw test history.
4. Scan user parquet files under prediction directory.
5. Build offline context only for test users that also have prediction parquet.

This shares the same split/cache/manifest data path as evaluator, avoiding reinterpretation of raw trajectory logic during metrics.

### 7.2 Channel merge

Before metric computation, offline metrics applies fixed channel merge via [resolve_channel_merges()](metrics/offline/channel_merge.py#L26) and [merge_channel_first_array()](metrics/offline/channel_merge.py#L51):

- `(0, 3)`
- `(1, 4)`

Rule: do element-wise `nanmean` between primary and secondary channel, write result back to primary channel, and zero out the secondary channel. Corresponding secondary feature rows in metric outputs are also zeroed by [zero_out_metrics_output()](metrics/offline/channel_merge.py#L74).

### 7.3 Ground truth and observed mask

For each prediction row, [slice_ground_truth()](metrics/offline/metric_core.py#L29) does:

```python
start = row["history_length"]
end = start + config.forecasting.forecasting_length
ground_truth = history[:, start:end]
ground_truth_observed_mask = np.isfinite(ground_truth)
```

If `history_length` is missing, not a non-negative integer, or `end` exceeds trajectory length, that row is skipped and counted.

All point/quantile metrics ignore missing positions using `ground_truth_observed_mask`. Missing ground truth is stored as `NaN` in output matrices.

### 7.4 Benchmark-level scale

Before calculating MASE/sQL, pipeline scans all actual prediction windows in that run and estimates scales from all valid seasonal pairs. Relevant implementation entrypoints are [accumulate_hour_of_day_scale_statistics()](evaluation/point_metrics.py#L151), [finalize_hour_of_day_scales()](evaluation/point_metrics.py#L83), [accumulate_global_scale_statistics()](evaluation/point_metrics.py#L265), and [finalize_global_scales()](evaluation/point_metrics.py#L118):

- Seasonal lag is fixed at 24 hours.
- Only positions where both `target_idx` and `target_idx - 24` are finite are used.
- `mase`/`sql` use scales by `(feature, hour-of-day)`, shape `(n_features, 24)`.
- `mase_all` uses one global scale per feature, shape `(n_features,)`.
- Entries with count 0 or scale 0 are set to `NaN`.

### 7.5 Metric definitions and output shapes

Each prediction row is passed to [MetricsComputer.compute()](metrics/offline/metric_core.py#L87), producing one metric matrix of shape `(n_features, forecasting_length)`:

- `mae = abs(pred - gt)`
- `mase = abs(pred - gt) / scale[feature, target_hour % 24]`, core function [compute_mase()](evaluation/point_metrics.py#L365).
- `ql`: standard pinball quantile loss, averaged over quantile dimension, core function [compute_ql()](evaluation/quantiles_metrics.py#L26).
- `sql = ql / scale[feature, target_hour % 24]`, core function [compute_sql()](evaluation/quantiles_metrics.py#L68).

If a model has no quantile predictions, offline code wraps point predictions as a single quantile. If quantile levels are missing, default levels are uniformly generated in `(0, 1)` based on quantile count.

Offline outputs are saved by metric directory using [save_metrics_result_by_metric()](metrics/offline/parquet_io.py#L72):

```text
results/metrics/<run_key>/
  config.yaml
  mae/<user_id>.parquet
  mse/<user_id>.parquet
  mase/<user_id>.parquet
  mase_all/<user_id>.parquet
  ql/<user_id>.parquet
  sql/<user_id>.parquet
```

Each metric parquet keeps only one metric column plus:

- `user_id`
- `model`
- `history_length`
- `forecasting_length`

Current skip policy: if `results/metrics/<run_key>/mae/<user_id>.parquet` already exists, that user is skipped for that run.

## 8. Aggregation and paper results

### 8.1 Hourly summary

[summary_metrics_result.py](metrics/deprecated/summary_metrics_result.py#L457) targets one specific metric directory, for example `results/metrics/<run_key>/mae`:

```bash
python src/forecasting_evaluation/metrics/deprecated/summary_metrics_result.py \
  --model seasonal_naive=results/metrics/seasonal_naive/mae \
  --output-dir results/metrics_summary
```

Outputs:

- `mae_by_model_channel_hour.csv`
- `statistical_result.csv`

It aggregates mean/std/n by `model, channel, channel_idx, hour`, and reports user count, sample count, and average inference time.

### 8.2 paper_result

[paper_result/](paper_result) is a second-stage aggregation layer for paper tables and figures. Common entrypoints include:

- [run_hour_group_metric_summary.py](../../scripts/forecasting/run_hour_group_metric_summary.py#L1)
- [plot_model_metric_combined.py](../../scripts/forecasting/plot_model_metric_combined.py#L578)
- [get_hour_group_metric_summary.slurm](../../jobs/imperial/slurm/forecasting/paper_result/get_hour_group_metric_summary.slurm#L1)
- [plot_model_metric_combined.slurm](../../jobs/imperial/slurm/forecasting/paper_result/plot_model_metric_combined.slurm#L1)

[hour_group_metric_summary.py](paper_result/hour_group_metric_summary.py#L438) reads `results/metrics/<model>/<metric>/<user>.parquet`, first performs NaN-aware averaging across multiple metric matrices for the same user, then aggregates by channel and 3-hour groups `0-2, 3-5, ..., 21-23`.

## 9. Common caveats

- `data.day_remain_mask` is mainly used to generate sample index; what evaluation truly requires is `data.sample_index_file`.
- `forecasting.daily_start_hour_offset` only affects runtime window slicing, not sample index, manifest, or cache key.
- The semantic meaning of `history_length` is now unified as forecast origin; do not treat it as fixed `n_steps` for PyPOTS.
- [SubTrajectoryGenerator](data/generator.py#L25) is still present in code, but evaluator main path has converged to `history_cf cache + row-group manifest`.
- `features.covariate_types` is currently interface-only; code does not actually build hour/day covariates.
- Zeros are converted to `NaN` first, so all downstream observed masks depend on finite values.
- If prediction parquet or metric parquet already exists, default skip policy may make reruns look like they did not update; check `overwrite_existing_parquet` or clean target outputs.
- Binary metrics are computed in a separate pipeline [binary_offline_calculate.py](metrics/binary_offline_calculate.py#L421); this document focuses on the main continuous forecasting metrics path.
