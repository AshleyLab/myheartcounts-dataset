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

- `mhc-forecast-eval` (Hydra CLI for reproducible public benchmark runs)
- `scripts/run_forecasting_eval.py` (legacy prediction-generation wrapper)
- `src/forecasting_evaluation/metrics/offline_calculate.py` (offline metric calculation)
- `src/forecasting_evaluation/metrics/paper_result_generator_all_channels.py` (appendix-style raw hour-group tables)
- `src/forecasting_evaluation/metrics/skill_score_summary.py` and `fairness_skill_score_summary.py` (paper scoring summaries)

---

## 0.5. Reproducible Runs via `mhc-forecast-eval`

Use the Hydra CLI when you want composable config presets, timestamped Hydra
run directories, multirun sweeps, and Sherlock dispatch.

Single run:

```bash
mhc-forecast-eval model=seasonal_naive
```

Multirun:

```bash
mhc-forecast-eval --multirun \
  model=seasonal_naive,seasonal_naive_average_history,autoARIMA,autoETS
```

The public Hydra config tree lives at `configs/forecasting/`:

- `model/`: `seasonal_naive`, `seasonal_naive_average_history`, `autoARIMA`,
  `autoETS`, `chronos2`, `toto`, `mixlinear`, `dlinear`, `segrnn`
- `data/`: trajectory dataset paths, split file, day mask, sample index
- `forecasting/`: horizon and daily start-hour offset
- `features/`: current 19-channel feature selection
- `output/`: prediction parquet output root and overwrite policy

### Checkpoints and Releases

Baseline and statistical models do not require checkpoints. Learned and
finetuned models can be launched either with a direct nested checkpoint path:

```bash
mhc-forecast-eval model=dlinear model.dlinear.checkpoint_path=/path/to/model.pypots
```

or with an imputation-style release directory:

```bash
mhc-forecast-eval model=dlinear model.release_dir=/path/to/openmhc-dlinear-forecast/
```

The release directory must contain `openmhc_manifest.json`. Forecasting accepts
the same core fields used by imputation releases:

```json
{
  "spec_version": 1,
  "kind": "dlinear",
  "checkpoint": "model.pypots",
  "normalization_stats": null,
  "arch": {"n_steps": 168, "n_pred_steps": 24, "n_features": 19},
  "provenance": {}
}
```

`kind` must match `model.type`. The CLI resolves `checkpoint`, copies the
manifest into the Hydra run directory, and applies matching `arch` keys onto
the selected nested model config.

### Sherlock

Sherlock scripts live under `jobs/sherlock/forecasting_eval/`.

```bash
# Full suite; chains summary aggregation by default.
jobs/sherlock/forecasting_eval/submit_all.sh

# Baselines only.
sbatch jobs/sherlock/forecasting_eval/run_baselines.sbatch

# Learned checkpoint example.
export MHC_FORECAST_DLINEAR_RELEASE_DIR=/path/to/openmhc-dlinear-forecast
sbatch jobs/sherlock/forecasting_eval/run_dlinear.sbatch
```

Common environment knobs:

| Variable | Meaning |
|---|---|
| `MHC_REPO_DIR` | Repo checkout on Sherlock; defaults to the script-resolved repo root |
| `MHC_VENV` | Virtualenv path; defaults to `/scratch/users/$USER/envs/mhc-benchmark` if present |
| `MHC_DATA_DIR` | Dataset root containing `hourly_trajectory`, `splits`, and `forecasting_sample_index`; Sherlock scripts prefer `~/.cache/openmhc/data-full` when unset |
| `MHC_FORECAST_RUNS_ROOT` | Output root; defaults to `results/forecasting_eval/sherlock` |
| `MHC_FORECAST_<MODEL>_RELEASE_DIR` | Release dir for `CHRONOS2`, `TOTO`, `MIXLINEAR`, `DLINEAR`, or `SEGRNN` |

To add a new Hydra model preset, add its dataclass fields in `config.py`,
register construction in `models/registry.py`, add a YAML preset under
`configs/forecasting/model/`, and include it in the Sherlock submission list if
it should run in the paper sweep.

### Full-Data Seasonal Naive Parity Check

The full paper forecasting sample is the default public config when
`MHC_DATA_DIR` points at the full OpenMHC cache. Expected test coverage is:

- `827` test users
- `43,563` test forecasting samples
- one prediction parquet per test user

The local full-data Seasonal Naive parity run used:

```bash
MHC_DATA_DIR=$HOME/.cache/openmhc/data-full HYDRA_FULL_ERROR=1 \
mhc-forecast-eval \
  model=seasonal_naive \
  data.num_workers=1 \
  experiment_name=FullSeasonalNaiveParity \
  output.results_dir=/tmp/mhc_forecast_full_seasonal_naive_final \
  output.overwrite_existing_parquet=true \
  hydra.run.dir=/tmp/mhc_forecast_full_seasonal_naive_final/hydra \
  hydra.job.chdir=false
```

This writes predictions to:

```text
{results_dir}/{experiment_name}/seasonal_naive/{user_id}.parquet
```

Seasonal Naive is the paper's `0` skill-score reference, so a
Seasonal-Naive-only run can verify coverage, raw metrics, and zero-skill
normalization against itself. It cannot reproduce the main table's average rank
or fairness-adjusted score without the other model outputs.

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
  type: seasonal_naive | seasonal_naive_average_history | autoARIMA | autoETS | chronos2 | toto | mixlinear | dlinear | segrnn
  name: str | null
  release_dir: str | null
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
    pretrained_model_name_or_path: str
    checkpoint_path: str | null
    training_output_dir: str | null
    finetuned_ckpt_name: str | null
    device: cuda | cpu
    torch_dtype: auto | float32 | float16 | bfloat16
  toto:
    pretrained_model_name_or_path: str
    checkpoint_path: str | null
    lora_alpha: float | null
    device: cuda | cpu
    context_length: int
    num_samples: int
    samples_per_batch: int
    use_kv_cache: bool
    time_interval_seconds: int
  mixlinear:
    checkpoint_path: str | null
    device: cuda | cpu
    batch_size: int
    n_steps: int | null
    n_pred_steps: int | null
    n_features: int | null
    period_len: int
    lpf: int
    alpha: float
    rank: int
  dlinear:
    checkpoint_path: str | null
    device: cuda | cpu
    batch_size: int
    n_steps: int | null
    n_pred_steps: int | null
    n_features: int | null
    moving_avg_window_size: int
    individual: bool
    d_model: int | null
  segrnn:
    checkpoint_path: str | null
    device: cuda | cpu
    batch_size: int
    n_steps: int | null
    n_pred_steps: int | null
    n_features: int | null
    seg_len: int
    d_model: int
    dropout: float
```

The canonical source of truth is `ForecastingModelConfig` in `config.py`; the
Hydra presets under `configs/forecasting/model/` set the common paper defaults.

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

For appendix raw hour-group tables, disable phone/watch channel merging:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths seasonal_naive_0=results/forecasting_eval/Exp/seasonal_naive \
  --metrics-output-path results/metrics_nocombine \
  --no-combine-channels
```

Arguments:

- `--evaluation-result-paths` (alias `--run-dirs`): one or more `name=path` mappings.
- `--metrics-output-path` (alias `--output-dir`): root output dir for offline metrics.
- `--max-user`: optional sequential cap on how many user parquet metrics files to compute per run.
- `--combine-channels` / `--no-combine-channels`: controls whether paired
  phone/watch step and distance channels are merged before metrics.

### 4.2 What It Does

For each run mapping:

1. Load `config.yaml` from forecasting model directory.
2. Rebuild dataset split and iterate unique users in test split.
3. Index prediction parquet files by user.
4. Re-extract history from trajectory and reconstruct GT by:
   - `start = history_length`
   - `end = history_length + forecasting_length`
5. Optionally merge paired phone/watch step and distance channels.
6. Compute:
   - `mae`: element-wise absolute error matrix `(feature, horizon)`
   - `mse`: element-wise Mean Squared Error matrix `(feature, horizon)`
   - `mase`: hour-of-day scaled absolute error matrix `(feature, horizon)`
   - `mase_all`: globally scaled absolute error matrix `(feature, horizon)`
   - `ql`: quantile loss matrix `(feature, horizon)`
   - `sql`: hour-of-day scaled quantile loss matrix `(feature, horizon)`
7. Flatten performance dict into `perf_*` columns.
8. Save per-user parquet into per-metric subdirectories.

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

### 4.4 Combined vs. No-Combine Metrics

There are two paper metrics layouts:

| Output root | Channel behavior | Intended use |
|---|---|---|
| `results/metrics` | Combines `(0, 3)` step count and `(1, 4)` distance, then zeroes the secondary channels | Main skill/fairness summaries |
| `results/metrics_nocombine` | Keeps all 19 channels separate | Appendix raw hour-group tables |

The default is combined metrics to preserve main-score behavior. Use
`--no-combine-channels` when regenerating appendix tables such as
`tab:hour_of_day_mae`, `tab:hour_of_day_ql`, and
`tab:hour_of_day_mae_stepcount`.

Example full-data no-combine metrics calculation from an existing Seasonal
Naive prediction run:

```bash
MHC_DATA_DIR=$HOME/.cache/openmhc/data-full \
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths 'Seasonal Naive=/tmp/mhc_forecast_full_seasonal_naive_final/FullSeasonalNaiveParity/seasonal_naive' \
  --metrics-output-path /tmp/mhc_forecast_full_seasonal_naive_final/metrics_nocombine \
  --no-combine-channels
```

Expected full-data result for Seasonal Naive:

- `827` per-user parquet files under each metric directory
- `43,563` saved rows
- `0` skipped rows

### 4.5 Raw Appendix Table Generation

Generate the appendix-style grouped raw table from no-combine metrics:

```bash
python -m forecasting_evaluation.metrics.paper_result_generator_all_channels \
  --models-json '{"Seasonal Naive":"/tmp/mhc_forecast_full_seasonal_naive_final/metrics_nocombine/Seasonal_Naive"}' \
  --output-dir /tmp/mhc_forecast_full_seasonal_naive_final/paper_raw_table_check \
  --output-file /tmp/mhc_forecast_full_seasonal_naive_final/paper_raw_table_check/seasonal_naive_raw_grouped_nocombine.csv
```

The active paper appendix is included from `paper/00_main.tex` via
`sections/appendix/forecasting.tex`. Important table labels:

- `tab:hour_of_day_mae`: raw continuous-channel MAE table, averaged across
  start times `0`, `6`, `12`, and `18`
- `tab:hour_of_day_ql`: raw continuous-channel QL table, averaged across those
  start times
- `tab:binary_model_channel_metrics`: raw binary-channel table
- `tab:hour_of_day_mae_stepcount`: start-time-specific Step Count MAE table

A default `mhc-forecast-eval model=seasonal_naive` run corresponds to
`Seasonal Naive(0)` in `tab:hour_of_day_mae_stepcount`. Matching the full
appendix raw tables exactly requires running start-time offsets `0`, `6`, `12`,
and `18`, computing no-combine metrics for each, and aggregating them together.

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
- `toto`
- `mixlinear`
- `dlinear`
- `segrnn`

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
Run forecasting prediction generation with Hydra:

```bash
mhc-forecast-eval model=seasonal_naive
```

Hydra override example:

```bash
mhc-forecast-eval \
  model=seasonal_naive \
  output.results_dir=results/forecasting_eval/dev \
  output.overwrite_existing_parquet=true
```

Run offline metrics calculation:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths ExpA=results/forecasting_eval/ExpA/20260324_120000 \
  --metrics-output-path results/metrics
```

For no-combine appendix metrics:

```bash
python src/forecasting_evaluation/metrics/offline_calculate.py \
  --evaluation-result-paths ExpA=results/forecasting_eval/ExpA/seasonal_naive \
  --metrics-output-path results/metrics_nocombine \
  --no-combine-channels
```

Run offline metrics summary:

```bash
python src/forecasting_evaluation/metrics/summary_metrics_result.py \
  --model ExpA=results/metrics/seasonal_naive/mae \
  --output-dir results/metrics_summary
```

---

## 8. Implementation Notes and Current Caveats

- Full-data runs rely on cache artifacts under `data/processed/forecasting_eval_h5/`.
  For non-PyPOTS/statistical models, evaluation and offline metrics now build
  only the raw `history_cf` cache needed for the test split. PyPOTS models still
  use the full train/validation/test cache bundle for training/evaluation.
- `ForecastingDataLoader.load_splits()` selects split rows by indexed
  `user_id` lookup instead of Hugging Face multiprocessing filters. This avoids
  materializing large trajectory rows just to split users and is important for
  full-data runs.
- Offline metrics are streamed by user. The pipeline computes global scale
  denominators first, then reads one user's prediction parquet rows, computes
  records, and writes that user's per-metric parquet files immediately.
- Hour-of-day/global scale denominators are accumulated with a vectorized bulk
  path over all start indices for a user. This preserves the scalar denominator
  semantics while avoiding tens of thousands of Python calls.
- Seasonal Naive empirical quantiles are vectorized across complete seasonal
  windows. This keeps the full 43,563-sample baseline run feasible while
  preserving the original loop behavior.
- `SubTrajectoryGenerator` requires user entries in `data.sample_index_file` and does not implement fallback candidate generation.
- `forecasting.valid_prediction_window.valid_day_threshold` and `valid_hour_threshold` are configured but not enforced in sample generation.
- `valid_prediction_window.history_length` is configured but current slicing still uses all history before `index_hours`.
