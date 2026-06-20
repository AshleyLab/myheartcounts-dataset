# Forecasting Evaluation (Track 3)

This package covers the forecasting track of the MyHeartCounts benchmark. It
evaluates how well a model forecasts the next `horizon` hours of multivariate
wearable sensor data for held-out users, given the full history up to each
forecast origin.

This organized as follows:

- **Part 1** — implement a custom forecaster and score it with
`openmhc.evaluate_forecasting`. Most users only need this.
- **Part 2** — the `mhc-forecast-eval` Hydra CLI for reproducible runs, and how
to add your own model to it.
- **Part 3** — reproduce the paper leaderboard (skill / rank / fairness) from
scratch.

How the pipeline works internally — the preprocessing chain,
`ForecastingEvaluator.run()`, the prediction parquet schema, and the offline
metric math — lives in [INTERNALS.md](INTERNALS.md). The full runtime config
schema is the dataclass tree in `[config.py](config.py)`.

---

## Dataset at a glance

Numbers below are for the default forecasting sample index
`sample_index_P_24_M_H_7_3_S_100.json` (24 h horizon; candidate days filtered by
missing-mask + the `H_7_3` historical check, capped at 100 windows/user). Each
**window** is one 24 h-ahead forecast anchored at a day boundary; its context is
the full trajectory prefix before that boundary. Window semantics are
model-agnostic — every model gets the same full prefix and owns any
truncation/padding (see [INTERNALS.md](INTERNALS.md#1-data-preprocessing)).

| Split      | Users     | Windows     | Windows/user (mean) |
| ---------- | --------- | ----------- | ------------------- |
| Train      | 1,621     | 83,510      | 51.5                |
| Validation | 289       | 16,061      | 55.6                |
| Test       | 827       | 43,563      | 52.7                |
| **Total**  | **2,737** | **143,134** | 52.3                |

Per-window context length (the full prefix handed to every model, before any
model-specific truncation):

|                             | mean  | median | p10 → p90  |
| --------------------------- | ----- | ------ | ---------- |
| Calendar span (days)        | 1,191 | 1,055  | 91 → 2,455 |
| Observed days (non-missing) | 389   | 197    | 21 → 1,042 |

Only ~30–37 % of each prefix carries actual observations; the rest is
missing/NaN.

---

## Part 1 — Custom forecasters via the public API (`openmhc`)

### Minimal example

```python
import numpy as np
import openmhc

class LastValueForecaster:
    """Repeat the last observed hour across the horizon."""

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        # history: (n_channels, history_length) — full prefix up to the origin
        last = history[:, -1:]                       # (n_channels, 1)
        return np.tile(last, (1, horizon)).astype(np.float32)

results = openmhc.evaluate_forecasting(
    LastValueForecaster(),
    version="full",            # "full" or "xs"; checked against the dataset marker
    forecasting_length=24,     # forecast horizon in hours
)
print(results.summary())       # per-channel metric table
results.to_csv("forecasting_results.csv")
```

`evaluate_forecasting(forecaster, version, forecasting_length=24, data_dir=None, seed=42, max_samples=None)` returns a `[ForecastingResults](../openmhc/_results.py)`
instance. The dataset root is resolved from `data_dir` first, then the
`MHC_DATA_DIR` environment variable; if neither is set the API raises.

### The `Forecaster` protocol

Any object with this method works (duck-typed, no base class required):

```python
def predict(
    self,
    history: np.ndarray,   # (n_channels, history_length) float32, full prefix; may contain NaN / be short
    horizon: int,          # number of future hours to predict
    *,
    variable_names: list[str] | None = None,        # optional, forwarded if declared
    past_covariates: dict[str, np.ndarray] | None = None,
    future_covariates: dict[str, np.ndarray] | None = None,
    index_days: np.ndarray | None = None,
) -> np.ndarray:           # (n_channels, horizon) float32 point forecast
```

- **Same window for every model.** Windows are selected by data-quality
criteria only, so all models see the identical set. The harness passes the
*full prefix* before the forecast origin; a model that wants a fixed context
truncates / pads it itself.
- **NaN → Seasonal-Naive fallback.** The harness never drops a window for
model-capability reasons. Emit `NaN` for any cell you cannot predict — the
harness substitutes the Seasonal-Naive baseline there before scoring and
reports how often via `results.overall_fallback_rate` / `results.fallback_rate`.
- **Quantiles are optional.** Return a `(point, quantiles)` tuple instead of a
bare array (`quantiles` shape `(n_channels, horizon, n_quantiles)`) and expose
the matching levels as a `quantile_levels` attribute. The benchmark ranks
point forecasts.
- **Optional metadata.** The harness inspects your signature once and forwards
only the keyword-only kwargs you declare (`variable_names`, `past_covariates`,
`future_covariates`, `index_days`). Three-argument forecasters work unchanged.

### Built-in reference forecasters (`openmhc.forecasters`)

All are fine-tuned / trained on the MHC training split and load a released
checkpoint bundle via `from_release(...)` (a local dir or an `hf://` URI):


| Class                                                          | Model                       | Extra                            |
| -------------------------------------------------------------- | --------------------------- | -------------------------------- |
| `Chronos2Forecaster`                                           | Fine-tuned Amazon Chronos-2 | `pip install 'openmhc[chronos]'` |
| `TotoForecaster`                                               | Fine-tuned Datadog Toto     | `pip install 'openmhc[toto]'`    |
| `DLinearForecaster`, `SegRNNForecaster`, `MixLinearForecaster` | From-scratch neural         | `pip install 'openmhc[pypots]'`  |


```python
from openmhc.forecasters import Chronos2Forecaster

fc = Chronos2Forecaster.from_release("hf://MyHeartCounts/openmhc-chronos2-fc")
results = openmhc.evaluate_forecasting(fc, version="full")
```

Released bundles live under `MyHeartCounts/openmhc-<model>-fc` on the Hugging
Face Hub.

### Results object

`ForecastingResults` exposes per-channel continuous (`mae`, `mase`, `ql`, `sql`)
and binary (`auroc`, `auprc`) metrics. Useful methods:

- `.summary()` — wide DataFrame, one row per channel, metrics as columns.
- `.to_dataframe()` — long-format DataFrame (includes per-channel `fallback_rate`).
- `.to_csv(path)` / `.to_json(path)` — dump results.
- `.to_submission_yaml(method_name=..., submitter_team=..., code_url=...)` —
render a paste-ready leaderboard submission.

---

## Part 2 — Reproducible runs & custom models via `mhc-forecast-eval` (Hydra)

Reach for the CLI (instead of the Part 1 Python API) when you want composable
YAML config presets, timestamped run directories with the resolved config and
release manifest copied in, Hydra `--multirun` sweeps, and SLURM dispatch. The CLI is declared in `[pyproject.toml](../../pyproject.toml)` as the`mhc-forecast-eval` console script; public-API do not need to touch Hydra.

### Configs

The config tree lives at `configs/forecasting/`, composed via `eval.yaml`:


| Group          | Picks                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model/`       | Forecaster preset + hyperparameters: `seasonal_naive`, `autoARIMA`, `autoETS`, `chronos2`, `toto`, `mixlinear`, `dlinear`, `segrnn` |
| `data/`        | Trajectory dataset paths, split file, day mask, sample index, workers                                                               |
| `forecasting/` | Forecast horizon and daily start-hour offset                                                                                        |
| `features/`    | Channel / covariate selection (the fixed 19-channel set)                                                                            |
| `output/`      | Prediction parquet root and overwrite policy                                                                                        |
| `metrics/`     | Which offline metrics to compute (point + pooled binary)                                                                            |


The schema is the dataclass tree in `[config.py](config.py)`
(`ForecastingEvalConfig`); Hydra validates every override against it.

### Usage

```bash
# Single run
mhc-forecast-eval model=seasonal_naive

# Multirun sweep
mhc-forecast-eval --multirun model=seasonal_naive,autoARIMA,autoETS

# Common overrides (any dataclass field is reachable via dotted keys)
mhc-forecast-eval \
  model=seasonal_naive \
  output.results_dir=results/forecasting_eval/dev \
  output.overwrite_existing_parquet=true
```

Each run writes one prediction parquet per test user **and** the offline metric
trees (point metrics per `channel × horizon` cell, plus pooled-within-user
binary metrics — see Part 3).

### Checkpoints and releases

Baseline and statistical models need no checkpoint. Learned / finetuned models
load either a direct checkpoint path or an imputation-style release directory:

```bash
mhc-forecast-eval model=dlinear model.dlinear.checkpoint_path=/path/to/model.pypots
# or
mhc-forecast-eval model=dlinear model.release_dir=/path/to/openmhc-dlinear-fc/
```

A release directory must contain `openmhc_manifest.json`:

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
manifest into the run directory, and applies matching `arch` keys onto the
selected nested model config.

### Adding a new forecaster to the CLI

Four edits. The model class uses the **same `predict` contract as the Part 1
protocol**, so a forecaster written for the public API drops straight in.

1. `**[config.py](config.py)`** — add the type to the `ModelType` literal,
  define a small `@dataclass` for its hyperparameters, and add a field on
   `ForecastingModelConfig`:
2. `**[models/registry.py](models/registry.py)**` — add an `elif` branch in
  `create_forecasting_model` that builds the model and sets `model_name`:
3. **The model class** — implement `predict(history, horizon)` (optionally
  subclass `[BasePredictionModel](models/base.py)` for the no-op `reset()` and
   the `model_name` / `quantile_levels` attributes the evaluator reads):
4. `**configs/forecasting/model/my_model.yaml`**:
  ```yaml
   # @package model
   type: my_model
   name: null
   release_dir: null
   my_model:
     window: 168
  ```

Run with `mhc-forecast-eval model=my_model`. The public-API path
(`openmhc.evaluate_forecasting(MyModel(...))`) keeps working in parallel — the
model class itself does not need to know about Hydra.

### SLURM Cluster submissions: 

SLURM submission scripts live under `jobs/sc-cluster/forecasting_eval/` (e.g.
`submit_all.sh`, `run_baselines.sbatch`). They read the dataset root from
`MHC_DATA_DIR` and each learned model's checkpoint from
`MHC_FORECAST_<MODEL>_RELEASE_DIR`. NOTE: the current files are configures for internal clusters but adaptation should be relatively straighforward for other SLURM-based HPC clusters.

---

## Part 3 — Reproduce the paper leaderboard from scratch

The canonical path is **raw dataset → predictions → offline metrics →
leaderboard**, all 12 models, no pre-existing artifacts, driven end-to-end by
one script.

**Step 1 — get the data and point `MHC_DATA_DIR` at it.**

```bash
python -c "import openmhc; openmhc.download_dataset(version='full')"

export MHC_DATA_DIR=/path/to/openmhc/data   # parent of hourly_trajectory/, splits/, forecasting_sample_index/
```

The full forecasting test split is **827 users / 43,563 samples** — use these
numbers as a sanity check. The `forecasting_sample_index` ships with the
packaged dataset; regenerating it from raw data is part of the (private)
preprocessing chain in [INTERNALS.md](INTERNALS.md#1-data-preprocessing). It has
no fallback path, so a missing index fails fast.

**Step 2 — provide checkpoints for the learned models.** Zeroshot foundation
models (`chronos2`, `toto`) and statistical models (`seasonal_naive`,
`autoARIMA`, `autoETS`) need none. The finetuned / trained entries
(`chronos2_finetuned`, `toto_finetuned`, `dlinear`, `mixlinear`, `segrnn`) need
release bundles — set each model's `release_dir` in
`configs/paper/sweep_forecasting.yaml` (see Part 2, "Checkpoints and releases").
To reproduce only the no-checkpoint subset, pass `--models`.

**Step 3 — confirm the scoring config** in
`configs/paper/sweep_forecasting.yaml` (current defaults):

```yaml
baseline: seasonal_naive
continuous_metrics: [mae]
binary_metrics: [auroc]            # scored binary metric: auroc | auprc | f1
within_user_aggregation: micro     # micro (default) | macro
bootstrap: { enabled: true, n_boot: 1000, seed: 42, fairness: true }
```

**Step 4 — run the whole pipeline** (Phase 0 inference → 1 discover → 2
skill/rank → 3 bootstrap/fairness):

```bash
python scripts/paper_results/forecasting/run_paper_pipeline.py \
  --sweep-config configs/paper/sweep_forecasting.yaml
```

Phase 0 runs `mhc-forecast-eval` per model, which writes the prediction parquets
**and** the offline metric trees — point metrics per `channel × horizon` cell
and the pooled-within-user binary metrics. This step is GPU-heavy for the
foundation / learned models and is normally dispatched as per-model SLURM jobs;
add `--skip-eval` to run only Phases 1–3 once the metric trees exist.

**Step 5 — read the leaderboard** in `output_root`:
`forecasting_skill_score_{long,model_summary,wide}.csv`,
`forecasting_grouped_metric_rank_{long,user_level_long,wide}.csv`,
`forecasting_fairness_skill_score.csv`, and the matching `*_bootstrap.csv`.

### Scoring semantics (brief)

- **Skill score** (vs `baseline`, default `seasonal_naive`): per task
`(channel, metric)`, take the paired ratio `E_model / E_baseline` over common
users, clip to `[0.01, 100]`, geometric-mean → `R`; `skill = 1 − R`.
Higher-is-better metrics (`auroc/auprc/f1`) are first converted to error
`e = 1 − value`.
- **Scopes — per-channel, 4 categories, one overall.** Every channel is scored
individually, then grouped into 4 sensor categories — **activity** (0–4),
**physiology** (5–6), **sleep** (7–8), **workout** (9–18) — and a single
**`overall`** that is *category-balanced*: it averages the 4 categories with equal
weight (each as one within-category geometric mean of log-ratios), so the 10
workout channels can't dominate the headline. Skill, rank, and fairness all share
this per-channel → category → overall shape. See [METRICS.md](METRICS.md).
- **Within-user aggregation** (default `micro`): continuous metrics pool all
finite horizon cells across a user's windows; binary metrics are already
pooled per user by the producer (so the toggle is a no-op for them).
- **Rank**: per channel, rank the models within each user, then average ranks over
users (mean-of-ranks, users-first); each sensor category and the `overall` headline
average those per-channel task ranks (overall = the 4 categories equally). Mirrors the
imputation track and the skill score's user-first collapse.
- **Fairness**: disparity-ratio fairness skill score across demographic subgroups,
macro-averaged over `age_group` + `sex`, and reported per channel, per category,
and as the category-balanced `overall`.
- **Bootstrap**: resample users (the cluster unit) with replacement and recompute
via the same readers → `mean / se / 95% CI`.

### Interpreting metrics

Main scores treat every channel **per task**: each channel (including the paired
phone/watch signals — steps `0,3`, distance `1,4`) keeps its own skill ratio, and
the category scopes combine their channels with a geometric mean — the same way
the imputation track aggregates channels. All 19 channels get their own
skill/rank/fairness cell; the 4 category scopes and the category-balanced
`overall` sit on top. Pass `--combine-channels` (legacy) to instead nan-mean the
paired signals before scoring, as used for some appendix tables. Binary-channel
metrics (sleep, workout) are AUROC/AUPRC/F1 pooled within each user. The math is
in [METRICS.md](METRICS.md) and
[INTERNALS.md](INTERNALS.md#7-offline-metric-computation).

> **Sparse per-channel fairness:** a per-channel or per-category fairness cell can
> be undefined or noisy when a binary channel has too few users in a subgroup — the
> disparity guards (≥2 common subgroups, baseline gap > 0) drop such tasks, so read
> single-channel fairness cells with their `n_boot` / CI width in mind.

> **Memory:** the binary-metric pass and the bootstrap load all users' contexts
> into RAM — run on a real allocation (~64–96 GB), not a small interactive
> shell, or it OOM-kills silently. Keep the prediction parquets if you may
> re-score binary metrics later: pooled AUROC needs every per-sample prediction
> score (re-paired at score time with cache-derived ground truth), which the
> per-window metric trees do not retain.

---

## Internals

For how the engine works — the data preprocessing chain, the step-by-step
`ForecastingEvaluator.run()` walkthrough, the per-window slicing and prediction
parquet schema, the offline metric definitions (`mae`, `mase`, `ql`, `sql`,
channel merge, hour-of-day scales), and current caveats — see
[INTERNALS.md](INTERNALS.md). The complete config schema is the dataclass tree in
`[config.py](config.py)`.