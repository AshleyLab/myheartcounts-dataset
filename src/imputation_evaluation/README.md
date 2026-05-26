# Imputation Evaluation (Track 2)

This package covers the imputation track of the MyHeartCounts benchmark. It evaluates how well a method reconstructs artificially masked sensor data across six masking scenarios, on minute-level daily samples of shape `(19 channels × 1440 minutes)`.

Most users should read **Part 1** and call `openmhc.evaluate_imputation` — they never need to import anything from `imputation_evaluation` directly. **Part 2** documents the library internals for developers hacking on the eval pipeline itself.

---

## Part 1 — Using the public API (`openmhc`)

### Minimal example

```python
import openmhc
from openmhc.imputers import MeanImputer

imputer = MeanImputer()                       # fits itself on the train split in __init__
results = openmhc.evaluate_imputation(
    imputer,
    masking_scenarios="all",                   # or a list of scenario names
    seed=42,
)
print(results.summary())                       # one row per (scenario, split, channel_group)
results.to_csv("imputation_results.csv")
```

`evaluate_imputation(imputer, masking_scenarios="all", data_dir=None, seed=42)` returns an [`ImputationResults`](../openmhc/_results.py) instance. Large benchmark payloads are resolved from `data_dir` first, then `MHC_DATA_DIR`. If neither is provided, the API raises instead of silently falling back to `~/.cache/openmhc/data`.

### The `Imputer` protocol

Any object with this method works (duck-typed, no base class required):

```python
def impute(
    self,
    data: np.ndarray,             # (N, 19, 1440) float32, NaN at all missing positions
    observed_mask: np.ndarray,    # (N, 19, 1440), 1 = originally observed, 0 = naturally missing
    target_mask: np.ndarray,      # (N, 19, 1440), 1 = positions to impute (subset of observed_mask)
    *,
    sample_indices: np.ndarray | None = None,   # (N,) split-local indices
    user_ids: list[str] | None = None,
    dates: list[str] | None = None,             # ISO "YYYY-MM-DD"
) -> np.ndarray:                  # (N, 19, 1440) float32, imputed at target_mask == 1
```

The optional keyword-only arguments are introspected from your signature ([adapter at `src/openmhc/_evaluate.py:615`](../openmhc/_evaluate.py)): three-arg implementations work unchanged, and personalized methods that declare `user_ids` / `dates` / `sample_indices` get them forwarded automatically. The harness never calls `fit` or any setup hook — do all setup in `__init__`.

### Built-in reference imputers (`openmhc.imputers`)

| Class | Module | Method |
|---|---|---|
| `MeanImputer` | `mean.py` | Per-channel global mean |
| `ModeImputer` | `mode.py` | Per-channel global mode (rounded) |
| `LinearImputer` | `linear.py` | Linear interpolation between observed anchors per sample/channel; NOCB/LOCF at boundaries |
| `LOCFImputer` | `locf.py` | Last observation carried forward; back-fills the left edge |
| `TemporalMeanImputer` | `temporal_mean.py` | Per-(channel, minute-of-day) mean — captures diurnal pattern |
| `TemporalModeImputer` | `temporal_mode.py` | Per-(channel, minute-of-day) mode |
| `PersonalizedMeanImputer` | `personalized.py` | Per-user per-channel mean, global fallback for unseen users |
| `PersonalizedModeImputer` | `personalized.py` | Per-user per-channel mode, global fallback |
| `PersonalizedTemporalMeanImputer` | `personalized.py` | Per-user diurnal pattern, global fallback |
| `TorchImputer` | `torch_wrapper.py` | Generic wrapper for a pre-trained `torch.nn.Module` (normalization + device + sigmoid for binary channels) |
| `BRITSImputer`, `TimesNetImputer`, `DLinearImputer`, `FEDformerImputer` | `pypots.py` | Wrappers around the published PyPOTS imputation models. Install `pip install 'openmhc[pypots]'`. See [`docs/neural-imputers.md`](../../docs/neural-imputers.md). |
| `LSM2Imputer`, `LSM2WeeklySparseImputer` | `lsm2.py` | Wrappers around the in-house masked-autoencoder ViT for 1D wearables (daily / weekly / weekly-sparse). Install `pip install 'openmhc[lsm2]'`. See [`docs/neural-imputers.md`](../../docs/neural-imputers.md). |
| `BaseImputer` | `_base.py` | Optional base class with channel-statistic helpers |

### Masking scenarios

`openmhc.list_masking_scenarios()` returns all 6 scenario names. They fall into two tiers:

**Tier 1 — Structural (sensor data collection issues)**

| Scenario | Description | Default config |
|---|---|---|
| `random_noise` | Non-overlapping random patches on individual channels. Simulates brief sensor noise or BT drops. | `patch_size=10`, `mask_ratio=0.8` |
| `temporal_slice` | Contiguous time blocks across all channels. Simulates device downtime (showering, charging). | `mask_ratio=0.5`, `min_block=30`, `max_block=60` |
| `signal_slice` | Drop entire channels for the day, either random channels or a whole device group (iPhone / Watch). | `mask_ratio=0.5`, groups `iphone=[0,1,2]`, `watch=[3,4,5,6]` |

**Tier 2 — Semantic (physiologically meaningful periods)**

| Scenario | Description | Applicability |
|---|---|---|
| `sleep_gap` | Mask all channels except the two sleep channels (7, 8) during detected sleep (asleep OR in-bed). | Days with sleep data |
| `workout_gap` | Mask Watch HR + Active Energy (ch 5–6) during detected workouts. | Days with workout data containing valid HR/AE |
| `intensity_failure` | Mask Watch HR + Active Energy when HR exceeds a threshold (default 160 BPM). | Days with high-intensity periods |

### Channels

Names come from `openmhc.SENSOR_CHANNELS`. The order matches column order in tensors.

| Idx | Channel | Type | Unit |
|---|---|---|---|
| 0 | `iphone_steps` | Continuous | steps/min |
| 1 | `iphone_distance` | Continuous | m/min |
| 2 | `iphone_flights` | Continuous | count/min |
| 3 | `watch_steps` | Continuous | steps/min |
| 4 | `watch_distance` | Continuous | m/min |
| 5 | `watch_hr` | Continuous | bpm |
| 6 | `watch_energy` | Continuous | cal/min |
| 7 | `sleep_asleep` | Binary | 0 / 1 |
| 8 | `sleep_inbed` | Binary | 0 / 1 |
| 9–18 | `workout_*` (walking, cycling, running, other, mixed_cardio, strength, elliptical, hiit, functional, yoga) | Binary | 0 / 1 |

### Metrics

Computed in `evaluation/metrics.py`.

- **Continuous channels (0–6):** per-channel `rmse`, `mae`, `mse`, plus the normalized variants divided by training std (`normalized_rmse`, `normalized_mae`, `normalized_mse`). Aggregated as `mean_normalized_rmse` / `mean_normalized_mae` / `mean_normalized_mse` under the `"continuous"` group.
- **Binary channels (7–18):** per-channel `balanced_accuracy` and `roc_auc`. Aggregated as `macro_balanced_accuracy` and `macro_roc_auc` under the `"binary"` group.

### Results object

`ImputationResults.scenarios` is `{scenario: {split: {group: {metric: value}}}}`. Useful methods:

- `.summary()` — wide DataFrame, one row per `(scenario, split, channel_group)` with metrics as columns. Filters to the `continuous` / `binary` aggregate groups.
- `.to_dataframe()` — long-format DataFrame including per-channel rows.
- `.to_csv(path)` / `.to_json(path)` — dump full results.
- `.to_submission_yaml(method_name=..., submitter_team=..., code_url=...)` — render a paste-ready leaderboard submission.

### Custom imputers

If you need access to the training data (to compute statistics, fit a model, etc.), the public helpers stream from the same DataLoader the eval harness uses:

```python
import openmhc

for data, mask in openmhc.iter_train_data():
    # data: (B, 19, 1440) float32, NaN at missing positions
    # mask: (B, 19, 1440) float32, 1 = observed
    ...

# Or any split:
for data, mask in openmhc.iter_split_data("val"):
    ...

# Lightweight metadata only (no tensors loaded):
meta = openmhc.load_sample_metadata("test")  # [{"sample_idx": 0, "user_id": ..., "date": ...}, ...]
```

All of these accept `data_dir=` / `seed=` overrides; defaults match `evaluate_imputation`.

---

## Part 1.5 — Reproducible runs via `mhc-impute-eval`

Reach for the CLI (instead of the Python API in Part 1) when you want:

- Composable YAML configs and CLI overrides instead of in-code wiring.
- A timestamped run directory with the resolved config and the loaded release manifest copied in.
- W&B logging out of the box (`wandb=on`).
- Hydra `--multirun` sweeps over methods / scenarios / data subsets.
- SLURM dispatch on Sherlock via the `submitit` launcher.

The CLI is declared in [`pyproject.toml`](../../pyproject.toml) as the
`mhc-impute-eval` console script. Public-API users (`openmhc.evaluate_imputation`)
never touch Hydra.

### Configs

Config presets live at `configs/imputation/` (repo root), composed via the
`defaults:` list in [`configs/imputation/eval.yaml`](../../configs/imputation/eval.yaml):

| Group | Presets | Picks |
|---|---|---|
| `data/` | `default`, `xs` | Data root, batch size, splits, multi-day window |
| `masking/` | `all_six` (default), `sleep_gap_only`, `workout_gap_only`, `random_noise_only` | Which masking scenarios are enabled |
| `method/` | `mean`, `mode`, `linear`, `locf`, `temporal_mean`, `temporal_mode`, `brits`, `timesnet`, `dlinear`, `fedformer`, `lsm2`, `lsm2_weekly_sparse` | Imputation method + arch / runtime kwargs |
| `output/` | `default` | `results_dir`, optional experiment name |
| `evaluation/` | `default` | `compute_metrics`, `save_pairs` |
| `visualization/` | `off`, `on` | Visualization toggle (see Known gaps below) |
| `sensitivity/` | `off`, `on` | Demographic subgroup metrics |
| `wandb/` | `off`, `on` | W&B logging |

The schema is the dataclass tree in [`config.py`](config.py) (`ImputationEvalConfig`); Hydra validates every override against it.

### Usage

```bash
# Reference imputer — fits on the train split inside its __init__
mhc-impute-eval method=mean

# Paper checkpoint — manifest-bundled release
mhc-impute-eval method=brits method.release_dir=path/to/openmhc-brits-paper/

# Sweep across methods and masking scenarios
mhc-impute-eval --multirun \
  method=brits,timesnet,dlinear,fedformer \
  method.release_dir=releases/${method.type} \
  masking=all_six,sleep_gap_only
```

Common overrides (anything on a dataclass is reachable via dotted keys):

| Override | Effect |
|---|---|
| `data=xs` | Use the small dev subset (downloadable via `openmhc.download_dataset(version="xs")`) |
| `method.device=cuda:0` | Inference device for neural imputers |
| `method.inference_batch_size=128` | Inference batch size for neural imputers |
| `masking=sleep_gap_only` | Restrict to one scenario (faster smoke tests) |
| `wandb=on wandb.tags='[smoke]'` | Log this run to W&B |
| `seed=7` | Override the top-level RNG seed |

### Paper-checkpoint manifests

For neural methods (`brits`, `timesnet`, `dlinear`, `fedformer`, `lsm2`,
`lsm2_weekly_sparse`), the recommended path is `method.release_dir=<dir>`. The
release dir contains an `openmhc_manifest.json` plus the checkpoint and
optional `normalization_stats.json`; the CLI reads the manifest, validates
that `kind` matches the requested method, and reconstructs the model via
`cls.from_release(...)`. The manifest is then copied into the run dir so every
result is traceable to its exact checkpoint + arch. See
[`docs/neural-imputers.md`](../../docs/neural-imputers.md) for the bundle
schema and the `tools/build_manifest.py` packager.

If you don't have a manifest (e.g. a bare PyPOTS file from your own training
run), the inline-arch fallback on each method YAML is consulted instead — see
the comments at the top of [`configs/imputation/method/brits.yaml`](../../configs/imputation/method/brits.yaml)
and [`configs/imputation/method/lsm2.yaml`](../../configs/imputation/method/lsm2.yaml).
Arch fields must match the trained model or PyPOTS's `load()` raises a
size-mismatch error.

### SLURM (Sherlock)

```bash
mhc-impute-eval --multirun hydra/launcher=sherlock_submitit \
  method=brits,timesnet,dlinear,fedformer method.release_dir=releases/${method.type}
```

The `sherlock_submitit` launcher YAML lives in the shared `eval_hydra` package
and is picked up via `hydra.searchpath: [pkg://eval_hydra.configs]` in
`eval.yaml`. Override partition / GPU count on the CLI as usual
(`hydra.launcher.partition=gpu hydra.launcher.gres=gpu:1`).

### Output layout

Single run:

```
${output.results_dir}/<YYYYMMDD_HHMMSS>_<method.type>/
├── results.json              # the run's metrics, JSON-serialized
├── .hydra/                   # resolved config + overrides
└── openmhc_manifest.json     # copied if method.release_dir was used
```

Multirun: subdirs under `${output.results_dir}/multirun/<ts>/<method.type>__<job_num>/`.

### Adding a new method to the CLI

Three edits:

1. **[`src/imputation_evaluation/config.py`](config.py)** — add your method name to the `MethodConfig.type` `Literal[...]`. If it needs hyperparameters, define a small `@dataclass MyMethodConfig` and add a field on `MethodConfig` (mirror `LSM2MethodConfig` / `PyPOTSMethodConfig`).
2. **[`src/imputation_evaluation/hydra/registry.py`](hydra/registry.py)** — register the class in `_REFERENCE_CLASSES` (no checkpoint) or `_PAPER_CHECKPOINT_CLASSES` (uses `ReleaseLoadableMixin.from_release`). For an exotic builder, write a small function that returns `(_ImputerMethodAdapter(imputer), manifest_or_None)`.
3. **`configs/imputation/method/my_method.yaml`**:

   ```yaml
   # @package method
   type: my_method
   # any new fields you added to MethodConfig
   ```

Run with `mhc-impute-eval method=my_method`. The public-API path
(`openmhc.evaluate_imputation(MyImputer())`) keeps working in parallel — the
imputer class itself does not need to know about Hydra.

---

## Part 2 — Library internals (`imputation_evaluation/`)

This package is the engine `openmhc.evaluate_imputation` calls. The public surface (the `Imputer` protocol, results object, dataset-path resolution) lives in `openmhc/`; this package contains the masking, evaluation, and I/O machinery.

### Layout (current truth)

```
src/imputation_evaluation/
├── __init__.py
├── config.py                  # All dataclasses (see "Configuration" below)
├── runner.py                  # run_eval() — library entry point
├── sensitivity.py             # Age/sex subgroup mapping
├── data/
│   ├── data_loader.py         # ImputationDataLoader (HF + splits + QA filters)
│   ├── mask_dataset.py        # PyTorch Dataset for parallel mask generation
│   └── splits.py              # User-level split utilities
├── masking/
│   ├── base.py, generator.py  # MaskGenerator protocol, MaskCacheGenerator
│   ├── random_noise.py, temporal_slice.py, signal_slice.py
│   ├── sleep_gap.py, workout_gap.py, intensity_failure.py
│   └── __init__.py            # create_mask_generators(...) registry
├── evaluation/
│   ├── evaluator.py           # ImputationEvaluator (main orchestrator)
│   ├── metrics.py             # compute_scenario_metrics, compute_per_sample_metrics
│   ├── pair_aggregator.py     # Re-aggregate metrics from saved (gt, pred) pairs
│   └── pair_writer.py         # Persist raw pairs to Parquet for offline analysis
└── io/
    ├── writer.py              # results.json / config.yaml writer
    └── wandb_logger.py        # Optional W&B logging
```

The package does **not** contain `methods/` or `visualization/` within `src/imputation_evaluation/`. Hydra config YAMLs live at `configs/imputation/` (repo root); see Part 1.5. The `ImputationMethod` interface that `run_eval` expects is satisfied by the `_ImputerMethodAdapter` at [`src/openmhc/_evaluate.py:586`](../openmhc/_evaluate.py), which bridges from the public `openmhc.Imputer` protocol.

### Library entry point

```python
from imputation_evaluation.runner import run_eval
results: dict = run_eval(cfg, method=adapter, subgroup_mappings=None)
```

`run_eval(config, method, *, subgroup_mappings=None)` ([`runner.py`](runner.py)) does everything `evaluate_imputation` does minus the W&B / disk-writer / visualization side effects:

1. Loads splits via `ImputationDataLoader`.
2. Builds the enabled mask generators via `create_mask_generators(config.masking)`.
3. Generates and caches masks with `MaskCacheGenerator`.
4. Calls `method.fit(train_loader)` (the adapter uses this to accumulate channel stds; user `Imputer`s never see it).
5. Builds eval-only DataLoaders restricted to indices that have at least one applicable mask.
6. Runs `ImputationEvaluator` for val + test, returns a dict.

`method` must satisfy the (internal) `ImputationMethod` interface: `name`, `channel_stds`, `fit(train_loader)`, `impute(data, original_masks, artificial_masks, **kwargs)`. If you're calling `run_eval` from Python with a user `Imputer`, instantiate `_ImputerMethodAdapter` from `openmhc._evaluate` and pass it through.

### Configuration (`config.py`)

All settings live on one root dataclass:

```python
@dataclass
class ImputationEvalConfig:
    seed: int = 42
    data: DataConfig            # daily_hf_dir, splits, batch_size, num_workers, n_days, filters, preprocessing
    masking: MaskingConfig      # mask_seed + per-scenario sub-configs (RandomNoiseConfig, …)
    method: MethodConfig        # type + nested MAE/PyPOTS configs (see Known gaps)
    output: OutputConfig        # results_dir, experiment_name
    evaluation: EvalConfig      # compute_metrics, save_pairs
    visualization: VisualizationConfig   # see Known gaps — currently no consumer
    sensitivity: SensitivityConfig       # demographic subgroup analysis (age + sex)
    wandb: WandbConfig          # optional W&B logging
```

Notable fields a dev would tune:

- `data.daily_hf_dir`, `data.split_file` — point at the dataset and the user-level split JSON.
- `data.max_samples_per_split` — fast smoke runs.
- `data.num_workers`, `data.num_eval_workers`, `data.num_eval_dl_workers` — see "Performance" below.
- `data.n_days` — multi-day context window (1–7). Defaults to 1.
- `masking.mask_seed` — controls the RNG used by every scenario.
- `masking.masks_file` — load pre-computed masks from `.npz` instead of regenerating, for like-for-like comparison across methods.
- `sensitivity.enabled` / `sensitivity.age_bins` — emit per-subgroup metrics. Demographics are looked up via [`src/labels/api.py`](../labels/api.py); samples with missing demographics group under `"unknown"`.

Per-scenario knobs are on the matching sub-config in `MaskingConfig` (`random_noise.patch_size`, `temporal_slice.min_block_size`, `intensity_failure.hr_threshold`, etc.) — see the defaults table in Part 1.

### Adding a masking scenario

1. Create a file in `masking/` implementing the protocol from `masking/base.py`:

   ```python
   from .base import MaskResult

   class MyNewMask:
       @property
       def name(self) -> str:
           return "my_new_mask"

       def generate(self, data, original_mask, rng):
           # data: (19, 1440); original_mask: (19, 1440) with 1=valid
           artificial_mask = np.zeros_like(original_mask)
           # ... fill in your logic ...
           # INVARIANT: artificial_mask can only be 1 where original_mask is 1.
           return MaskResult(artificial_mask=artificial_mask, applicable=True)
   ```

2. Add a config dataclass in `config.py` and a field on `MaskingConfig`.
3. Register the constructor in `masking/__init__.py:create_mask_generators`.
4. Add the scenario name to `openmhc._constants.MASKING_SCENARIOS` and toggle it in `openmhc._evaluate.evaluate_imputation` so it's reachable from the public API.

### Performance & memory

- **Bit-packed masks (32× compression).** Stored once per scenario; only the slice needed for the current batch is unpacked inside each worker (see `data/mask_dataset.py`).
- **Batch-by-batch eval.** Each batch is loaded once and run through all enabled scenarios before the next batch loads — minimizes DataLoader overhead.
- **`num_workers`** — DataLoader prefetching for data loading, mask generation, and `method.fit`.
- **`num_eval_workers`** — `ProcessPoolExecutor` for batch-level eval parallelism. Each worker handles all scenarios for its batch; memory scales with `num_eval_workers × batch_size`, not dataset size.
- **`num_eval_dl_workers`** — DataLoader workers used during eval; decouples from `num_workers` so prefetching can overlap with parallel evaluation.
- **Incremental statistics.** `_ImputerMethodAdapter.fit` accumulates per-channel stds in a single pass, never materializing the full train tensor.
- **Precision.** float32 inputs; continuous accumulators use float64 for stability; binary storage uses int8 (gt) + float16 (pred).

### Known gaps

These are real issues in the published code; documenting honestly rather than papering over.

- **`VisualizationConfig` is dead code.** It's defined in `config.py:236` and accepted by `run_eval`, but the `visualization/` package that would consume it was not published. Setting `visualization.enabled = True` has no effect today. Plot your own results from the saved Parquet pairs (`pair_writer.py`) if you need qualitative inspection.
- **`MethodConfig.type` literal includes `"lsm2"`, `"lsm2_weekly_sparse"`, and `"pypots"`** (LSM2 was formerly called MAE in the private companion repo). The matching imputer wrappers ship in `openmhc.imputers` (`LSM2Imputer`, `LSM2WeeklySparseImputer`, `BRITSImputer` / `TimesNetImputer` / `DLinearImputer` / `FEDformerImputer`). For other neural imputers, use `openmhc.imputers.TorchImputer` with your own `torch.nn.Module`.
- **Dangling type-only imports.** `runner.py:31` and `evaluation/evaluator.py:32` reference `imputation_evaluation.methods.base.ImputationMethod` under `TYPE_CHECKING`. The module doesn't exist; nothing breaks at runtime, but static type-checkers will complain. A future cleanup should either restore the protocol module or replace it with a `typing.Protocol` defined locally.
