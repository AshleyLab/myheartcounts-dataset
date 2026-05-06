# Forecasting Evaluation

This module implements trajectory-level multivariate forecasting evaluation on held-out users.

Current scope in code:

- Load trajectory data from Hugging Face disk format.
- Split by user into train/validation/test (evaluation runs on test split only).
- Build sub-trajectories from pre-generated per-user day indices.
- Run one configured forecasting model per task.
- Persist predictions to parquet.
- Compute offline metrics from saved prediction outputs.
- Summarize metrics across models/channels/horizons.

Main entrypoints:

- `scripts/run_forecasting_eval.py` (prediction generation)
- `src/forecasting_evaluation/metrics/offline_calculate.py` (offline metric calculation)
- `src/forecasting_evaluation/metrics/summary_metrics_result.py` (offline metric summary export)

---

## 1. Execution Flow (Code-Aligned)

Core orchestrator: `ForecastingEvaluator` in `evaluation/evaluator.py`.

`ForecastingEvaluator.run()` executes:

1. Print resolved config.
2. Load dataset and user splits via `ForecastingDataLoader.load_splits()`.
3. Create run directory and save run config via `PublicWriter`.
4. Run configured model on test trajectories (sequential only).
5. For each `(model, trajectory)`:
   - extract features using `MultivariateFeatureExtractor.extract()`
   - generate sub-trajectories using `SubTrajectoryGenerator.generate()`
   - call `model.predict_wrapper(sub_traj)`
   - append one prediction record per sub-trajectory with `PredictResultWriter.append()`
6. Finalize run output.

Important behavior in current implementation:

- Only `test_ds` is evaluated.
- One prediction parquet file is maintained per `(model_name, user_id)`.
- Evaluator does not compute metrics online.
- `SubTrajectoryGenerator` requires a precomputed sample index and has no fallback generation path.

---

## 2. Runtime Config

Root schema: `ForecastingEvalConfig` in `config.py`.

Top-level fields:

- `seed`
- `experiment_name`
- `debug_mode`
- `time_granularity`
- `data`
- `forecasting`
- `model`
- `features`
- `evaluator`
- `output`

Model configuration is single-model:

```yaml
model:
  type: seasonal_naive | seasonal_naive_average_history | autoARIMA | autoETS | chronos2
  name: str | null
  seasonal_naive:
    season_length: int
    quantile_levels: list[float]
  seasonal_naive_average_history:
    season_length: int
  autoARIMA:
    start_p: int
    start_q: int
    max_p: int
    max_q: int
    seasonal: bool
    start_P: int
    start_Q: int
    max_P: int
    max_Q: int
    max_d: int
    max_D: int
    information_criterion: str
    suppress_warnings: bool
    trace: bool
    error_action: str
    stepwise: bool
    n_jobs: int
    max_history_length: int | null
  autoETS:
    auto: bool
    sp: int
    information_criterion: str
    n_jobs: int
    max_history_length: int | null
  chronos2:
    temp: int
```

Core nested fields:

```yaml
data:
  trajectory_hf_dir: str
  task_name: str
  split_file: str | null
  day_remain_mask: str | null
  sample_index_file: str
  train_ratio: float
  val_ratio: float
  split_seed: int
  num_workers: int
  max_samples: int | null

forecasting:
  forecasting_length: int

features:
  covariate_types: list[hour_in_day | day_in_week] | null
  channel: all

evaluator:
  mode: sequential

output:
  results_dir: str
  save_config: bool
  overwrite_existing_parquet: bool
```

---

## 3. Input/Output Contracts

### 3.1 Trajectory Input (from HF dataset)

Required fields used in pipeline:

- `user_id` (str)
- `values` (array-like, shape `(T, C)`)
- `channel_names` (list[str], length `C`)

Additional required preprocessing artifact:

- `data.sample_index_file`: required JSON mapping `user_id -> [day indices]`
- Generate it before evaluation with `scripts/precompute_forecasting_inputs.py`

Important note:

- Forecasting evaluation does not support missing `sample_index_file`
- There is currently no fallback sampling path in `SubTrajectoryGenerator`
- If the file is missing, runtime validation will fail fast and instruct you to generate it first

### 3.2 Feature Extractor Output

`MultivariateFeatureExtractor.extract()` returns:

```python
{
  "history": np.ndarray,               # (n_features, T)
  "variable_names": list[str],
  "past_covariates": dict[str, np.ndarray],
  "future_covariates": dict[str, np.ndarray],
  "static_covariates": dict[str, object] | None,
}
```

Channel selection is controlled by `features.channel`:

- `all` -> fixed 19-channel feature set

### 3.3 Sub-Trajectory Output

`SubTrajectoryGenerator.generate()` yields `SubTrajectoryInput` with:

- `history`: `(n_features, index_hours)`
- `ground_truth`: `(n_features, forecasting_length)`
- `variable_names`
- optional covariates
- `index_days`
- `prediction_hours`

Slicing logic:

- `index_hours = index_days * 24 + daily_start_hour_offset`
- `history = series[:, :index_hours]`
- `ground_truth = series[:, index_hours:index_hours + forecasting_length]`

`daily_start_hour_offset` is a runtime-only forecasting setup parameter. It
does not change `sample_index_file`, cache/storage layout, or persisted
manifests; it only shifts the effective slice boundary when samples are cut
from loaded trajectories.

### 3.4 Prediction Parquet Row

`PredictResultWriter` stores one parquet file per `(model_name, user_id)`:

`{results_dir}/{experiment_name}/{model_name}/{user_id}.parquet`

Special rule for test experiments:

- If `experiment_name` starts with `Test`, model directory becomes
  `{model_name}_{YYYYMMDDHHMM}`.

Overwrite rule:

- `output.overwrite_existing_parquet = false` (default): if `{user_id}.parquet` already exists,
  evaluator skips this user.
- `output.overwrite_existing_parquet = true`: evaluator rewrites existing `{user_id}.parquet`.

One row per sample, with columns:

- `user_id`
- `model`
- `history_length`
- `point_predictions` (nested list or null)
- `quantile_predictions` (nested list or null)
- `performance` (dict, e.g. prediction time and memory usage)

---

## 4. Offline Metrics Calculate

Entry script: `src/forecasting_evaluation/metrics/offline_calculate.py`

### 4.1 CLI

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths runA=results/forecasting_eval/ExpA/seasonal_naive \
                            runB=results/forecasting_eval/ExpB/chronos2 \
  --metrics-output-path results/metrics
```

Arguments:

- `--evaluation-result-paths` (alias `--run-dirs`): one or more `name=path` mappings.
- `--metrics-output-path` (alias `--output-dir`): root output dir for offline metrics.
- `--max-user`: optional sequential cap on how many user parquet metrics files to compute per run.

### 4.2 What It Does

For each run mapping:

1. Load `config.yaml` from forecasting model directory.
2. Rebuild dataset split and iterate unique users in test split.
3. Index prediction parquet files by user.
4. Re-extract history from trajectory and reconstruct GT by:
   - `start = history_length`
   - `end = history_length + forecasting_length`
5. Compute:
   - `mae`: element-wise absolute error matrix `(feature, horizon)`
   - `mse`: element-wise Mean Squared Error matrix `(feature, horizon)`
   - `mase`: hour-of-day scaled absolute error matrix `(feature, horizon)`
   - `mase_all`: globally scaled absolute error matrix `(feature, horizon)`
   - `ql`: quantile loss matrix `(feature, horizon)`
   - `sql`: hour-of-day scaled quantile loss matrix `(feature, horizon)`
6. Flatten performance dict into `perf_*` columns.
7. Save per-user parquet into per-metric subdirectories.

If quantile levels are absent in prediction rows, levels are auto-generated as evenly spaced values.

### 4.3 Output Layout

```text
results/metrics/
  {model_name}/
    config.yaml
    mae/
      {sanitized_user_id}.parquet
    mse/
      {sanitized_user_id}.parquet
    mase/
      {sanitized_user_id}.parquet
    mase_all/
      {sanitized_user_id}.parquet
    ql/
      {sanitized_user_id}.parquet
    sql/
      {sanitized_user_id}.parquet
```

Each per-metric parquet row includes:

- `user_id`
- `model`
- `history_length`
- `forecasting_length`
- exactly one metric column from `{mae, mse, mase, mase_all, ql, sql}`
- optional `perf_*` scalar columns

Current skip policy:

- Metrics are skipped per user if `mae/{user_id}.parquet` already exists for the target model.
- Invalid rows (for example missing history length or out-of-range slice) are skipped and counted.

---

## 5. Offline Metrics Summary

Entry script: `src/forecasting_evaluation/metrics/summary_metrics_result.py`

### 5.1 CLI

```bash
python src/forecasting_evaluation/metrics/summary_metrics_result.py \
  --model SeasonalNaive=results/metrics/SeasonalNaive/mae \
  --model Chronos2=results/metrics/Chronos2/mae \
  --output-dir results/metrics_summary \
  --max-user 200 \
  --random-seed 42
```

Arguments:

- `--model`: repeatable mapping in format `MODEL_NAME=/path/to/one_metric_dir`.
  For example, use `results/metrics/<model_name>/mae` when summarizing `mae`.
- `--output-dir`: output folder for summary CSV files.
- `--max-user`: optional random user cap per model (for quick small-scale analysis).
- `--random-seed`: random seed for user sampling.

### 5.2 Summary Outputs

The script writes two CSV files:

1. `mae_by_model_channel_hour.csv`
2. `statistical_result.csv`

`mae_by_model_channel_hour.csv` columns:

- `model`
- `channel`
- `channel_idx`
- `hour`
- `mae_mean`
- `mae_std`
- `n`

`statistical_result.csv` columns:

- `model`
- `user_count`
- `sample_count`
- `avg_prediction_time_seconds`

Implementation notes:

- Aggregation is streaming and does not materialize a full long MAE table.
- User IDs are collected from parquet rows.
- Channel names are inferred from run config (`features.channel`) when available; otherwise fallback names are used.
- Because metrics are now saved into separate per-metric directories, this summary script should be pointed at the directory for the metric being summarized.

---

## 6. Supported Model Types

Supported `model.type` values:

- `seasonal_naive`
- `seasonal_naive_average_history`
- `autoARIMA`
- `autoETS`
- `chronos2`

If `name` is missing, evaluator uses `model.type` as the model name.

---

## 7. Quick Start

### 7.1 Data Preparation

Recommend:
```bash
dvc pull data/hourly_trajectory.dvc
dvc pull data/forecasting_sample_index.dvc
```

Or if you have raw HDF5 data, you can generate above data with scripts, reference: `jobs/imperial/pbs/forecasting/build_hourly_trajectory.pbs` and `jobs/imperial/pbs/forecasting/build_sample_index.pbs`.



### 7.2 Run evaluation
Run forecasting prediction generation:

```bash
python scripts/run_forecasting_eval.py \
  --config configs/forecasting_eval/default.yaml
```

Layered config override:

```bash
python scripts/run_forecasting_eval.py \
  --config configs/forecasting_eval/default.yaml \
  --config configs/forecasting_eval/test.yaml
```

Run offline metrics calculation:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths ExpA=results/forecasting_eval/ExpA/20260324_120000 \
  --metrics-output-path results/metrics
```

Run offline metrics summary:

```bash
python src/forecasting_evaluation/metrics/summary_metrics_result.py \
  --model ExpA=results/metrics/seasonal_naive/mae \
  --output-dir results/metrics_summary
```

---

## 8. Current Caveats

- `SubTrajectoryGenerator` requires user entries in `data.sample_index_file` and does not implement fallback candidate generation.
- `forecasting.valid_prediction_window.valid_day_threshold` and `valid_hour_threshold` are configured but not enforced in sample generation.
- `valid_prediction_window.history_length` is configured but current slicing still uses all history before `index_hours`.
