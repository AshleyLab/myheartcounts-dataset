# Imputation Evaluation Module

This module evaluates how well different imputation methods reconstruct artificially masked sensor data across various masking scenarios. It operates on **minute-level daily data** where each sample is a single day `(19 channels × 1440 minutes)` from the daily HuggingFace dataset.

## Quick Start

### End-to-end pipeline

The fastest way to go from a method to paper-ready results is the pipeline script, which chains evaluation, aggregation, registration, and paper metric computation:

```bash
# Full pipeline for a single method
python scripts/run_imputation_pipeline.py \
    --config configs/imputation_eval/base.yaml \
    --config configs/imputation_eval/methods/mae.yaml

# Multiple methods end-to-end
python scripts/run_imputation_pipeline.py \
    --config configs/imputation_eval/base.yaml \
    --methods mean locf temporal_mean

# With sensitivity analysis
python scripts/run_imputation_pipeline.py \
    --config configs/imputation_eval/base.yaml \
    --methods mean locf \
    --sensitivity

# Re-run aggregation + paper metrics on existing results (skip eval)
python scripts/run_imputation_pipeline.py \
    --skip-eval \
    --experiment-dir results/imputation_eval/my_experiment/ \
    --sensitivity

# Dry run: print commands without executing
python scripts/run_imputation_pipeline.py \
    --config configs/imputation_eval/base.yaml \
    --methods mean \
    --dry-run
```

Pass extra arguments through to `run_imputation_eval.py` after `--`:

```bash
python scripts/run_imputation_pipeline.py \
    --config configs/imputation_eval/base.yaml \
    --config configs/imputation_eval/methods/mae.yaml \
    -- --masking.mask_seed 99 --method.mae.device cpu
```

See `python scripts/run_imputation_pipeline.py --help` for all options.

### Running evaluation only

```bash
# Run with default config (mean imputation)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml

# Quick test run with limited samples
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --config configs/imputation_eval/test.yaml

# Parallel evaluation (4 workers for batch-level parallelism)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --data.num_eval_workers 4

# With visualization enabled (generates plots for qualitative assessment)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --visualization.enabled true \
    --visualization.plots_per_scenario 5

# With sensitivity analysis (demographic subgroup breakdown)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --sensitivity.enabled true

# Override settings via CLI
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --masking.mask_seed 123 \
    --data.max_samples_per_split 500
```

## Module Structure

```
src/imputation_evaluation/
├── config.py                       # Configuration dataclasses
├── sensitivity.py                  # Demographic subgroup mapping (age, sex)
├── data/
│   ├── data_loader.py              # Load daily HF dataset, apply splits
│   ├── mask_dataset.py             # PyTorch Dataset for parallel mask generation
│   └── splits.py                   # User-level split utilities
├── masking/
│   ├── base.py                     # MaskGenerator protocol, MaskResult
│   ├── random_noise.py             # Random patch masking
│   ├── temporal_slice.py           # Contiguous time block masking
│   ├── signal_slice.py             # Channel/device dropout
│   ├── sleep_gap.py                # Mask during sleep periods
│   ├── workout_gap.py              # Mask HR + AE during workouts
│   └── intensity_failure.py        # Mask during high HR
├── methods/
│   ├── base.py                     # ImputationMethod protocol
│   ├── mean_imputation.py          # Per-channel global mean
│   ├── linear_interpolation.py     # Per-channel linear interpolation
│   ├── locf.py                     # Last Observation Carried Forward
│   ├── mae_imputation.py           # MAE masked autoencoder
│   └── pypots_imputation.py        # PyPOTS models (BRITS, SAITS, TimesNet, FEDformer, ...)
├── evaluation/
│   ├── evaluator.py                # Main orchestrator
│   └── metrics.py                  # RMSE, MAE, KS, Wasserstein, Balanced Accuracy, ROC AUC
├── visualization/
│   ├── plotter.py                  # Visualization functions (method-agnostic)
│   └── __init__.py
└── io/
    └── writer.py                   # Results output (JSON, YAML)
```

## Masking Scenarios

The module provides 6 masking scenarios organized into two tiers:

### Tier 1: Structural Masks

These simulate common sensor data collection issues:

| Scenario | Description | Config |
|----------|-------------|--------|
| `random_noise` | Random non-overlapping patches of `patch_size` contiguous minutes on individual channels. Patches are placed greedily in random order until `mask_ratio` is reached. Simulates brief sensor noise or Bluetooth drops. | `patch_size=10`, `mask_ratio=0.8` |
| `temporal_slice` | Contiguous time blocks across ALL channels. Simulates device downtime (showering, charging). | `mask_ratio=0.5`, `min_block=30`, `max_block=60` |
| `signal_slice` | Drop entire channels for the day. Mode A: random channels. Mode B: entire device group (iPhone/Watch). | `mask_ratio=0.5` |

### Tier 2: Semantic Masks

These test reconstruction during physiologically meaningful periods:

| Scenario | Description | Applicability |
|----------|-------------|---------------|
| `sleep_gap` | Mask all channels except sleep channels (7, 8) during detected sleep (asleep OR inbed > 0). | Only days with sleep data |
| `workout_gap` | Mask HR + Active Energy (ch 5-6) during detected workouts. | Only days with workout data containing valid HR/AE |
| `intensity_failure` | Mask HR + Active Energy when heart rate exceeds threshold (default 160 BPM). | Only days with high-intensity periods |

## Channel Reference

The daily data has 19 channels:

| Index | Channel | Type | Unit |
|-------|---------|------|------|
| 0 | iPhone Steps | Continuous | steps/min |
| 1 | iPhone Distance | Continuous | m/min |
| 2 | iPhone Flights Climbed | Continuous | count/min |
| 3 | Watch Steps | Continuous | steps/min |
| 4 | Watch Distance | Continuous | m/min |
| 5 | Watch Heart Rate | Continuous | bpm |
| 6 | Watch Active Energy | Continuous | cal/min |
| 7 | Sleep: Asleep | Binary | 0/1 |
| 8 | Sleep: In Bed | Binary | 0/1 |
| 9-18 | Workout Types | Binary | 0/1 |

## Metrics

### Continuous Channels (0-6)
- **RMSE**: Root Mean Squared Error (global, per-channel)
- **MAE**: Mean Absolute Error (global, per-channel)
- **Mean Per-Sample RMSE**: Average of per-sample RMSE values.
- **Mean Per-Sample MAE**: Average of per-sample MAE values.
- **Mean Per-Sample KS Statistic**: Average of per-sample KS statistics comparing ground truth vs imputed distributions at masked positions. This replaces the global KS statistic to reduce memory usage.
- **Mean Per-Sample Wasserstein Distance**: Average of per-sample 1-Wasserstein (Earth Mover's) distances between ground truth and imputed distributions at masked positions. Scale-sensitive (same units as channel). Lower is better.
- **Mean Normalized RMSE**: Average of `(channel_rmse / channel_std_train)` across channels.
- **Mean KS Statistic**: Macro-average of per-channel "Mean Per-Sample KS Statistic" values.
- **Mean Wasserstein Distance**: Macro-average of per-channel mean per-sample Wasserstein distances across continuous channels.

### Binary Channels (7-18)
- **Balanced Accuracy**: Per-channel and macro-average
- **ROC AUC**: Per-channel and macro-average

## Configuration

### DataConfig

```yaml
data:
  daily_hf_dir: data/hf_daily      # Path to HF dataset
  split_file: null                  # Optional: JSON file with user splits
  train_ratio: 0.6                  # Train split ratio
  val_ratio: 0.1                    # Validation split ratio
  split_seed: 42                    # Seed for random splits
  max_samples_per_split: null       # Limit samples per split (null = no limit)

  # DataLoader workers (for data loading, mask generation, method fitting)
  batch_size: 5000                  # Samples per batch
  num_workers: 4                    # DataLoader worker processes
  pin_memory: true                  # Pin memory for faster GPU transfer

  # Evaluation parallelism (separate from DataLoader)
  # Each worker processes ALL 6 scenarios for its batch (batch-level parallelism)
  num_eval_workers: 1               # Parallel processes for batch eval (1 = sequential)
  num_eval_dl_workers: null          # DataLoader workers for eval (null = use num_workers)

  preprocessing:
    zero_to_nan: true               # Convert HR=0 and dead AE to NaN
  filters:
    min_wear_fraction: 0.5          # Remove days with <50% wear-time
    variance_filter_enabled: true   # Remove low-variance days
    variance_thresholds: null       # Uses defaults from hf_config.py
```

### MaskingConfig

```yaml
masking:
  mask_seed: 42                     # Seed for mask generation
  masks_file: null                  # Optional: load pre-computed masks
  random_noise:
    enabled: true
    patch_size: 10                  # Minutes per patch
    mask_ratio: 0.8                 # Fraction of valid data to mask
  temporal_slice:
    enabled: true
    mask_ratio: 0.5
    min_block_size: 30              # Minimum block in minutes
    max_block_size: 60              # Maximum block in minutes
  signal_slice:
    enabled: true
    mask_ratio: 0.5
    device_groups:
      iphone: [0, 1, 2]
      watch: [3, 4, 5, 6]
  sleep_gap:
    enabled: true
    asleep_channel: 7
    inbed_channel: 8
  workout_gap:
    enabled: true
    mask_channels: [5, 6]                   # Only mask HR + Active Energy
    workout_channels: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  intensity_failure:
    enabled: true
    hr_channel: 5
    hr_threshold: 160.0             # BPM threshold
    hr_unit: auto                   # auto-detect Hz vs BPM
    mask_channels: [5, 6]           # Only mask HR + Active Energy
```

### MethodConfig

```yaml
method:
  type: mean                        # "mean", "mode", "linear", "locf", "mae", or "pypots"
  decimal_precision: 1              # Rounding precision for mode computation

  # MAE-specific (only used when type: mae)
  mae:
    checkpoint_path: path/to/mae.ckpt  # or "wandb:ENTITY/PROJECT/ARTIFACT:VERSION"
    device: cuda
    inference_batch_size: 128

  # PyPOTS-specific (only used when type: pypots)
  pypots:
    model_path: models/pypots/brits  # Path to saved PyPOTS model directory
    model_name: brits                # Model class: brits, saits, timemixer, timemixerpp, fits, dlinear, timesnet, fedformer, trmf
    device: cuda
    inference_batch_size: 64
    # Architecture params (must match training config for the chosen model)
    n_steps: 1440
    n_features: 19
    rnn_hidden_size: 128             # BRITS only
    n_layers: 2                      # TimesNet, FEDformer
    top_k: 5                         # TimesNet
    d_model: 64                      # TimesNet, FEDformer
    d_ffn: 64                        # TimesNet, FEDformer
    n_kernels: 6                     # TimesNet
    n_heads: 4                       # FEDformer
    moving_avg_window_size: 25       # FEDformer
    dropout: 0.1                     # TimesNet, FEDformer
    apply_nonstationary_norm: false  # TimesNet
    version: Fourier                 # FEDformer
    modes: 32                        # FEDformer
    mode_select: random              # FEDformer
    trmf_lags: [1, 2, 3, 4, 5]      # TRMF lag indices
    trmf_K: 10                       # TRMF rank
    trmf_lambda_f: 0.1               # TRMF factor regularization
    trmf_lambda_x: 0.1               # TRMF coefficient regularization
    trmf_lambda_w: 0.1               # TRMF lag weight regularization
    trmf_alpha: 0.01                 # TRMF temporal update rate
    trmf_eta: 1.0                    # TRMF temporal regularization strength
    trmf_max_iter: 1000              # TRMF max EM iterations
```

### EvalConfig

```yaml
evaluation:
  include_ks: true                  # Compute KS statistic for continuous channels
                                    # Set false to reduce memory (no continuous value storage)
  include_wasserstein: true        # Compute per-sample Wasserstein distance for continuous channels
```

### OutputConfig

```yaml
output:
  results_dir: results/imputation_eval
  experiment_name: null             # Auto-generated if null
  save_config: true
```

### VisualizationConfig

```yaml
visualization:
  enabled: false                    # Set true to generate visualization plots
  plots_per_scenario: 5             # Number of random samples per masking scenario
  channels: null                    # Channels to plot (null = default 0-8: continuous + sleep)
  figsize_per_channel: [15.0, 2.5]  # Figure size (width, height) per channel subplot
  seed: null                        # Seed for sample selection (null = use main seed)
  dpi: 150                          # DPI for saved figures
  format: png                       # Output format: png, pdf, svg
  split: test                       # Which split to visualize: "val" or "test"
```

### SensitivityConfig

```yaml
sensitivity:
  enabled: false                    # Set true for demographic subgroup analysis
  age_bins: [18, 30, 40, 50, 60]   # Age bin edges -> "18-29", "30-39", "40-49", "50-59", "60+"
```

When enabled, metrics are broken down by **age group** and **biological sex** alongside the overall metrics. This reveals whether imputation quality differs across demographic subgroups. Demographics are looked up via the Labels API (`src/labels/api.py`) using `user_id` and `date` from each sample. Samples with missing demographics are grouped as `"unknown"`.

**Visualization Output**: When enabled, generates multi-channel time-series plots for randomly sampled days per scenario, showing:
- **Observed data** (gray) — unmasked regions showing true observations
- **Ground truth** (green) — true values in masked regions
- **Imputed values** (blue dashed) — model predictions in masked regions
- **Masked regions** (red shading) — highlights where imputation occurred

This provides qualitative assessment of imputation quality beyond aggregate metrics. The visualization is method-agnostic and works with any `ImputationMethod` implementation.
```

## Output Files

After running, results are saved to `{results_dir}/{experiment_name}/`:

```
results/imputation_eval/imputation_mean_20240115_120000/
├── results.json          # Per-scenario metrics for val/test
├── config.yaml           # Full configuration used
├── masks/                # Persisted masks for reproducibility
│   ├── val/
│   │   ├── random_noise.npz
│   │   ├── temporal_slice.npz
│   │   └── ...
│   └── test/
│       ├── random_noise.npz
│       └── ...
└── plots/                # Visualization plots (if enabled)
    ├── random_noise/
    │   ├── sample_000123.png
    │   ├── sample_000456.png
    │   └── ...
    ├── temporal_slice/
    │   └── ...
    └── ...
```

### results.json Structure

```json
{
  "config": {
    "method": "mean",
    "seed": 42,
    "mask_seed": 42
  },
  "scenarios": {
    "random_noise": {
      "val": {
        "n_samples": 1234,
        "continuous": {
          "mean_normalized_rmse": 0.85,
          "mean_ks_statistic": 0.12,
          "n_channels": 7
        },
        "binary": {
          "macro_balanced_accuracy": 0.65,
          "macro_roc_auc": 0.72,
          "n_channels": 12
        },
        "per_channel": {
          "ch_0": {"rmse": 15.2, "mae": 12.1, "normalized_rmse": 0.82, "ks_statistic": 0.10, "n_masked": 50000},
          "ch_7": {"balanced_accuracy": 0.68, "roc_auc": 0.75, "n_masked": 30000}
        },
        "subgroups": {
          "age_group": {
            "18-29": {"n_samples": 200, "continuous": {"mean_normalized_rmse": 0.83}, "binary": {...}, "per_channel": {...}},
            "30-39": {"n_samples": 400, "continuous": {"mean_normalized_rmse": 0.86}, ...},
            "60+":   {"n_samples": 100, ...},
            "unknown": {"n_samples": 34, ...}
          },
          "sex": {
            "male":    {"n_samples": 620, "continuous": {"mean_normalized_rmse": 0.84}, ...},
            "female":  {"n_samples": 580, ...},
            "unknown": {"n_samples": 34, ...}
          }
        }
      },
      "test": { ... }
    },
    "sleep_gap": { ... }
  }
}
```

> **Note**: The `"subgroups"` key is only present when `sensitivity.enabled: true`. Each subgroup contains the same metric structure as the overall results (continuous, binary, per_channel).
```

## Visualization

The module provides optional visualization of imputation results to complement quantitative metrics with qualitative assessment. When enabled via `visualization.enabled: true`, the system generates time-series plots for randomly sampled days per masking scenario.

### Plot Components

Each visualization shows a multi-channel view of a single day with:

1. **Observed Data (Gray)**: True sensor values in unmasked regions
2. **Ground Truth (Green)**: True values in artificially masked regions (what we're trying to reconstruct)
3. **Imputed Values (Blue Dashed)**: Model predictions for masked regions
4. **Masked Regions (Red Shading)**: Highlights where imputation occurred

### Usage Examples

```bash
# Generate 5 plots per scenario on test split (default)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --visualization.enabled true

# Generate 10 plots per scenario on validation split
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --visualization.enabled true \
    --visualization.plots_per_scenario 10 \
    --visualization.split val

# Visualize specific channels (HR, Active Energy, Sleep)
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --visualization.enabled true \
    --visualization.channels "[5, 6, 7, 8]"

# High-resolution PDF output
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --visualization.enabled true \
    --visualization.format pdf \
    --visualization.dpi 300
```

### Implementation Details

- **Method-agnostic**: Works with any `ImputationMethod` implementation (mean, linear, neural, etc.)
- **Random sampling**: Samples are selected randomly from applicable indices per scenario using configurable seed
- **Per-scenario output**: Plots are organized into subdirectories by masking scenario
- **Efficient**: Only loads data needed for visualization (doesn't affect evaluation memory usage)
- **Reusable functions**: Core plotting functions (`plot_imputation_channel`, `plot_imputation_sample`) are available for custom analyses

### Interpretation Tips

- **Close alignment** between green (ground truth) and blue dashed (imputed) indicates good reconstruction
- **Temporal patterns**: Check if imputation preserves temporal structure (e.g., circadian rhythms, activity patterns)
- **Boundary effects**: Look at imputation quality at edges of masked regions
- **Scenario-specific behavior**: Different scenarios reveal different failure modes (e.g., long gaps vs. short patches)

## Mask Subsampling (Per-User Day Cap)

For ablation studies that vary the number of days per user, `scripts/subsample_masks.py` creates reduced mask directories from existing masks. The eval pipeline runs unchanged — it just points to the new mask directory.

### How it works

1. Loads the daily HF dataset with the same filter chain used during mask generation (WearTimeFilter, LowChannelVarianceFilter)
2. Resolves each split-local index to `(user_id, date)` via the HF dataset
3. For each scenario's `.npz` file, groups applicable indices by user_id
4. Per user: if they exceed `max_days_per_user`, subsamples with a deterministic per-user seed (`seed + md5(user_id)`)
5. Writes new `.npz` files with the reduced applicable set

### Usage

```bash
# Generate subsampled masks (cap at 30 days per user)
python scripts/subsample_masks.py \
    --masks_dir data/imputation/masks/sharable_users_seed42_2026/ \
    --split_file data/splits/sharable_users_seed42_2026.json \
    --daily_hf_dir data/processed/daily_hf \
    --max_days_per_user 30 \
    --seed 42 \
    --output_dir data/imputation/masks/sharable_users_seed42_2026_max30days/

# Run eval unchanged, pointing to subsampled masks
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --config configs/imputation_eval/methods/mae.yaml \
    --masking.masks_file data/imputation/masks/sharable_users_seed42_2026_max30days/
```

### Output NPZ format

Output files are backward compatible with `ScenarioMasks.load()`, which only reads the three standard keys. Two metadata arrays are added for traceability:

| Key | Description | Used by pipeline |
|-----|-------------|-----------------|
| `indices` | Global sample indices (standard) | Yes |
| `masks_packed` | Bit-packed masks, uint8 (standard) | Yes |
| `shape` | `(N_applicable, C, T)` (standard) | Yes |
| `user_ids` | Per-sample user_id strings | No (metadata) |
| `dates` | Per-sample date strings | No (metadata) |

```python
# Inspect a subsampled NPZ
d = np.load("output_masks/val/random_noise.npz", allow_pickle=True)
print(list(d.keys()))       # ['indices', 'masks_packed', 'shape', 'user_ids', 'dates']
print(d["shape"])           # [N_kept, 19, 1440]
print(d["user_ids"][:3])    # ['user_001', 'user_001', 'user_002']
print(d["dates"][:3])       # ['2024-03-01', '2024-03-02', '2024-03-05']
```

### Key properties

- **Per-scenario independent**: each scenario's applicable set is subsampled separately, preserving evaluation power for rare scenarios (e.g., `intensity_failure`)
- **Deterministic**: uses `hashlib.md5` for per-user seeding (stable across Python sessions, unlike `hash()`)
- **No pipeline changes needed**: extra NPZ keys are ignored by `np.load()` / `ScenarioMasks.load()`

### Post-hoc ablation (alternative)

`scripts/ablation_days_per_user.py` provides a post-hoc alternative that operates on saved Parquet pairs from a completed eval run. It recomputes metrics at various day caps with bootstrap CIs, without re-running the pipeline. Use this for quick exploratory analysis; use mask subsampling for definitive results.

## Reproducibility

Masks are deterministically generated using `np.random.default_rng(seed)` and **always persisted** to disk. This ensures:

1. **Same masks across runs**: Running with the same config produces identical masks
2. **Fair method comparison**: Different imputation methods can be evaluated on identical masks by setting `masking.masks_file` to a previous run's masks directory

```bash
# First run: generates and saves masks (with parallel evaluation)
python scripts/run_imputation_eval.py \
    --config default.yaml \
    --data.num_eval_workers 4

# Second run with different method: reuses exact same masks
python scripts/run_imputation_eval.py \
    --config default.yaml \
    --method.type mode \
    --data.num_eval_workers 4 \
    --masking.masks_file results/imputation_eval/imputation_mean_20240115/masks/
```

## Imputation Methods

### Mean Imputation (`type: mean`)
Fills each artificially masked position with the global per-channel mean computed from training data. Ignores temporal structure entirely.

### Mode Imputation (`type: mode`)
Fills each artificially masked position with the per-channel mode (most frequent value after rounding) from training data. Suited for binary/categorical channels.

### Linear Interpolation (`type: linear`)
For each sample and channel, linearly interpolates between known (unmasked) observations along the time axis. Uses `np.interp`, which naturally handles boundaries:
- **Left boundary** (before first known observation): NOCB — fills with the first known value.
- **Right boundary** (after last known observation): LOCF — fills with the last known value.
- **Fallback**: if a channel has zero known observations in a sample, fills with the global channel mean from training data.

This method leverages temporal ordering and generally outperforms global-statistic baselines on scenarios with temporal structure (e.g., `temporal_slice`, `random_noise`).

### LOCF Imputation (`type: locf`)
Last Observation Carried Forward. For each sample and channel, carries the most recent known value forward to fill masked positions:
- **Forward fill**: each masked position is filled with the last known value before it.
- **Left boundary** (before first known observation): NOCB — back-fills with the first known value.
- **Fallback**: if a channel has zero known observations in a sample, fills with the global channel mean from training data.

LOCF is a simple temporal baseline that preserves level but not trend. It performs well when values change infrequently (e.g., binary channels, step-like signals).

### MAE Imputation (`type: mae`)
Uses a pre-trained Masked Autoencoder to reconstruct masked positions via encoder-decoder inference. Requires a trained MAE checkpoint (`.ckpt`). Applies instance normalization matching the training preprocessing, runs inference in batches on GPU, and denormalizes predictions. See `configs/imputation_eval/methods/mae.yaml`.

The `checkpoint_path` accepts both local paths and W&B artifact references. Artifact references are automatically downloaded and cached to `~/.cache/openmhc/artifacts/`:

```yaml
method:
  type: mae
  mae:
    # Local path
    checkpoint_path: results/mae/mhc-mae-ssl/1tms4zet/checkpoints/best.ckpt

    # Or W&B artifact reference (auto-downloaded + cached)
    checkpoint_path: "wandb:MHC_Dataset/mhc-mae-ssl/mae:latest"
```

Version aliases: `:latest` (most recent), `:v0`/`:v1`/... (specific version), or custom aliases set in the W&B UI.

You can also override via CLI without editing the config:

```bash
python scripts/run_imputation_eval.py \
    --config configs/imputation_eval/base.yaml \
    --config configs/imputation_eval/methods/mae.yaml \
    --method.mae.checkpoint_path "wandb:MHC_Dataset/mhc-mae-ssl/mae:latest"
```

See [docs/wandb_artifacts.md](../../docs/wandb_artifacts.md) for the full artifact workflow (uploading, versioning, aliases).

### PyPOTS Imputation (`type: pypots`)
Uses pre-trained PyPOTS models for imputation. Requires a model directory saved by PyPOTS during training (see `scripts/train_pypots.py`). The adapter handles the transpose between our channels-first format `(N, C, T)` and PyPOTS time-first format `(N, T, C)`. PyPOTS manages its own internal batching during inference.

Supported models: **BRITS**, **SAITS**, **TimeMixer**, **TimeMixerPP**, **FITS**, **DLinear**, **TimesNet**, **FEDformer**, **TRMF** (CPU-only).

**Training workflow:**
1. Install: `pip install -e .[pypots]`
2. Train: `python scripts/train_pypots.py --config configs/pypots/<model>.yaml`
3. Evaluate: `python scripts/run_imputation_eval.py --config configs/imputation_eval/pypots_<model>.yaml`

Available training configs: `brits.yaml`, `timemixer.yaml`, `timesnet.yaml`, `fedformer.yaml`, `trmf.yaml`.
Available eval configs: `pypots_brits.yaml`, `pypots_timesnet.yaml`, `pypots_fedformer.yaml`, `pypots_trmf.yaml`.

The training script exports data to H5 files (cached), then trains via PyPOTS's own training loop. Data exports reuse `ImputationDataLoader` to guarantee identical QA filters and user-level splits. See `src/pypots_training/` for the training package and `configs/pypots/GPU_MEMORY.md` for batch size guidance.

## Adding New Imputation Methods

1. Create a new file in `methods/` (e.g., `methods/neural_imputation.py`)
2. Implement the `ImputationMethod` protocol:

```python
from torch.utils.data import DataLoader

class NeuralImputation:
    @property
    def name(self) -> str:
        return "neural"

    @property
    def channel_stds(self) -> np.ndarray | None:
        return self._channel_stds

    def fit(self, train_loader: DataLoader) -> None:
        # Iterate over DataLoader batches
        for batch_idx, (data, masks) in enumerate(train_loader):
            # data: (B, 19, 1440) tensor
            # masks: (B, 19, 1440) tensor, 1=valid
            data = data.numpy()
            masks = masks.numpy()
            # ... accumulate statistics or train model ...

    def impute(
        self,
        data: np.ndarray,
        original_masks: np.ndarray,
        artificial_masks: np.ndarray,
    ) -> np.ndarray:
        # data: (N, 19, 1440) with NaN at artificial_mask==1 positions
        # Returns: (N, 19, 1440) with imputed values
        ...
```

3. Register in `methods/__init__.py`:

```python
def create_imputation_method(config: MethodConfig) -> ImputationMethod:
    if config.type == "mean":
        return MeanImputation()
    elif config.type == "neural":
        return NeuralImputation(config.neural)  # Add config fields as needed
    else:
        raise ValueError(f"Unknown method: {config.type}")
```

4. Update `config.py` with any new config fields
5. Add YAML config in `configs/imputation_eval/`

## Adding New Masking Scenarios

1. Create a new file in `masking/` implementing the `MaskGenerator` protocol:

```python
from .base import MaskResult

class MyNewMask:
    @property
    def name(self) -> str:
        return "my_new_mask"

    def generate(
        self,
        data: np.ndarray,           # (19, 1440)
        original_mask: np.ndarray,  # (19, 1440), 1=valid
        rng: np.random.Generator,
    ) -> MaskResult:
        # INVARIANT: artificial_mask can only be 1 where original_mask is 1
        artificial_mask = np.zeros_like(original_mask)
        # ... your masking logic ...
        applicable = True  # Set False if scenario doesn't apply
        return MaskResult(artificial_mask=artificial_mask, applicable=applicable)
```

2. Add config dataclass in `config.py`
3. Register in `masking/__init__.py`

## Performance & Memory

The module balances memory efficiency with speed:

- **PyTorch DataLoader**: Multi-worker data loading with memory-mapped HuggingFace datasets. Only accessed samples are loaded into RAM.
- **Configurable parallelism**: Adjust `num_workers`, `pin_memory`, and `prefetch_factor` via config for optimal performance on your hardware.
- **`num_workers`**: DataLoader worker processes for data loading, mask generation, and imputation method fitting. Standard PyTorch DataLoader parallelism.
- **`num_eval_workers`**: Parallel processes for batch-level evaluation using ProcessPoolExecutor. Each worker processes ALL 6 scenarios for its assigned batch independently. Batches are streamed to workers as they're loaded (not collected upfront), so memory scales with `num_eval_workers × batch_size` rather than total dataset size.
- **`num_eval_dl_workers`**: DataLoader workers for eval splits, independent of `num_workers`. Defaults to `num_workers` when null. Allows data prefetching to run concurrently with parallel evaluation.
- **Selective mask unpacking**: Masks are stored bit-packed (32x compression). During evaluation, only the masks needed for the current batch are unpacked, avoiding full-array decompression in each worker process.
- **Batch-by-batch evaluation**: Each batch is loaded once and evaluated for ALL masking scenarios before moving to the next batch. This minimizes data loading overhead.
- **Vectorized mask generation**: Structural mask generators (random_noise, temporal_slice, signal_slice) use vectorized numpy operations instead of Python loops.
- **Incremental statistics**: Training statistics are computed incrementally across batches without loading all data into memory.
- **Precision**: Input data is loaded as float32. Continuous metric accumulators use float64 for numerical stability when summing across large sample counts. Binary channel storage uses int8 for ground truth (0/1) and float16 for predictions.
- **`max_samples_per_split`**: Set this to a small value (e.g., 1000-10000) for quick test runs during development. See `configs/imputation_eval/test.yaml`.

## Dependencies

- `numpy`: Array operations
- `datasets`: HuggingFace dataset loading (memory-mapped)
- `torch`: PyTorch DataLoader, Dataset, and ZeroToNaNTransform preprocessing
- `sklearn`: Balanced accuracy, ROC AUC metrics
- `matplotlib`: Visualization plots (optional, only needed if `visualization.enabled: true`)
- `jsonargparse`: CLI argument parsing
- `pyyaml`: Config serialization
- `h5py`: H5 export for PyPOTS training
- `pypots`: PyPOTS imputation models (optional, install via `pip install -e .[pypots]`)
