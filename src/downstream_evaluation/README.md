# Downstream Evaluation Pipeline

Unified sklearn-based evaluation pipeline for health classification/regression tasks across multiple feature extraction methods and temporal granularities.

## Overview

The pipeline supports two evaluation pathways:

1. **Feature store** (stat_simple, stat_full, ssl_encoder, multirocket, jets_encoder):
   Extract features once -> store -> aggregate to user level -> sklearn classifier

2. **Supervised sequence** (gru_d, brits):
   Train per-task model on raw time series -> predict -> aggregate probabilities

Both pathways support **weekly** (168h) and **daily** (24h) segment types.

## Precomputed Artifacts

The pipeline depends on these precomputed artifacts:

| Artifact | Path | Build Command |
|----------|------|---------------|
| weekly_labels_lookup | `data/processed/weekly_labels_lookup.parquet` | `python scripts/labels/build_labels_lookup.py` |
| daily_labels_lookup | `data/processed/daily_labels_lookup.parquet` | `python scripts/labels/build_labels_lookup.py --segment-type daily` |
| weekly_hf | `data/processed/weekly_hf/` | `python -m data.processing.daily_hourly_hf_to_weekly_hf` |
| daily_hourly_hf | `data/processed/daily_hourly_hf/` | `python -m data.processing.daily_hf_to_daily_hourly_hf` |
| split_file | `data/splits/sharable_users_seed42_2026.json` | `python scripts/create_split.py` |
| clip_dates | `data/labels/clip_dates.json` | Static (temporal filtering cutoffs) |

Labels lookups are **index-aligned** with their respective HF datasets and must be rebuilt whenever the underlying dataset changes.

## Directory Structure

```
src/downstream_evaluation/
├── config.py                    # Dataclass-based configuration schema
├── feature_store.py             # WeekFeatureStore: extract-once, reuse features
├── README.md                    # This file
├── data/
│   ├── aggregation.py           # Segment-to-user pooling (mean, coverage-weighted)
│   ├── data_loader.py           # HF dataset loading, label attachment, daily_hourly_hf prep
│   └── splits.py                # User-based train/val/test splitting
├── evaluation/
│   ├── evaluator.py             # Main orchestrator
│   ├── metrics.py               # AUROC, AUPRC, Accuracy, F1, MSE, MAE, Pearson/Spearman R, QWK
│   ├── global_score.py          # Type-balanced GlobalScore aggregation
│   └── proxy_gs.py              # Proxy GlobalScore for fast HPO
├── feature_extractors/
│   ├── base.py                  # FeatureExtractor Protocol
│   ├── baseline_extractor.py    # Mean/std features (38-dim stat_simple, 1056-dim stat_full)
│   ├── encoder_extractor.py     # SSL encoder embeddings (256-dim)
│   ├── multirocket_extractor.py # MultiRocket features (~50K-dim)
│   └── jets_triplet_extractor.py # JETS observation-level features
├── supervised_models/
│   ├── base.py                  # Supervised model protocol
│   ├── data_prep.py             # Data preparation for sequence models
│   ├── grud_model.py            # GRU-D (PyPOTS) single-task
│   ├── brits_model.py           # BRITS (PyPOTS) single-task
│   ├── multitask_grud_model.py  # GRU-D multi-task (shared encoder)
│   └── multitask_brits_model.py # BRITS multi-task (shared encoder)
├── models/
│   └── registry.py              # Classifier factory with RobustStandardScaler
└── io/
    └── writer.py                # JSON/CSV output writer
```

## Usage

### Config Structure

```
configs/downstream_eval/
├── base.yaml            # Shared settings (data paths, classifiers, aggregation)
├── stat_simple.yaml     # Statistical baseline (mean/std, 38-dim)
├── stat_full.yaml       # Statistical full features (1056-dim)
├── ssl_encoder.yaml     # SSL encoder embeddings (256-dim)
├── multirocket.yaml     # MultiRocket features (~50K-dim)
├── jets_encoder.yaml    # JETS triplet encoder features
├── gru_d.yaml           # GRU-D supervised sequence model
├── brits.yaml           # BRITS supervised sequence model
├── gru_d_multitask.yaml # GRU-D multi-task
├── brits_multitask.yaml # BRITS multi-task
└── temporal_windows.yaml # Temporal windowing experiments
```

### Basic Usage

```bash
# Statistical baseline (mean/std, 38-dim)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml

# Statistical full features (1056-dim)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_full.yaml

# SSL encoder features (256-dim, requires GPU + checkpoint)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/ssl_encoder.yaml

# MultiRocket features (~50K-dim)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/multirocket.yaml

# JETS triplet encoder features
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/jets_encoder.yaml

# GRU-D supervised sequence model
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/gru_d.yaml

# BRITS supervised sequence model
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/brits.yaml
```

### Daily Segment Type

Use `--data.segment_type daily` to evaluate on daily (24h) segments from
`daily_hourly_hf` instead of weekly (168h) segments. This works for all
feature store methods (stat_simple, stat_full, multirocket) and supervised
sequence models (gru_d, brits).

```bash
# Daily evaluation
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml \
    --data.segment_type daily

# Daily with specific tasks
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml \
    --data.segment_type daily \
    --experiment.tasks "[Diabetes, BiologicalSex]"
```

When `segment_type=daily`, the pipeline:
- Loads from `daily_hourly_hf_dir` instead of `weekly_hf_dir`
- Transposes values/mask from (19, 24) channels-first to (24, 19) time-first
- Restores NaN where mask==1 (daily_hourly_hf is zero-filled)
- Uses `daily_labels_lookup.parquet` for labels
- Skips the `min_valid_days_per_week` coverage filter (not applicable)

### CLI Overrides

```bash
# Run specific tasks only
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml \
    --experiment.tasks "[Diabetes, Hypertension, BiologicalSex]"

# Specific time windows
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml \
    --experiment.time_windows "full,before_label,before_10w"

# Resume from previous run (skip completed tasks)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/stat_simple.yaml \
    --experiment.resume true

# W&B artifact for SSL checkpoint (auto-downloaded + cached)
python scripts/downstream_eval/run_downstream_eval.py \
    --config configs/downstream_eval/base.yaml \
    --config configs/downstream_eval/ssl_encoder.yaml \
    --features.ssl_encoder.checkpoint_path "wandb:MHC_Dataset/mhc-ssl/encoder:latest"
```

## Data Configuration

```yaml
data:
  segment_type: weekly            # "weekly" (168h) or "daily" (24h)
  weekly_hf_dir: data/processed/weekly_hf
  daily_hourly_hf_dir: data/processed/daily_hourly_hf
  weekly_labels_lookup_path: data/processed/weekly_labels_lookup.parquet
  daily_labels_lookup_path: data/processed/daily_labels_lookup.parquet
  split_file: data/splits/sharable_users_seed42_2026.json
  clip_dates_path: data/labels/clip_dates.json
  min_valid_days_per_week: 5      # Filter weeks with < N valid days (0=off, skipped for daily)
```

The `active_labels_lookup_path` property automatically routes to the correct
labels parquet based on `segment_type`.

## Feature Types

| Feature Type | Config | Dimensions | Method |
|-------------|--------|------------|--------|
| stat_simple | `stat_simple.yaml` | 38 | Per-channel mean + std |
| stat_full | `stat_full.yaml` | 1056 | 28 statistical features x (19 values + 19 masks) |
| ssl_encoder | `ssl_encoder.yaml` | 256 | Pretrained encoder embeddings |
| multirocket | `multirocket.yaml` | ~50,000 | Random convolutional kernels |
| jets_encoder | `jets_encoder.yaml` | 256 | JETS triplet observation embeddings |
| gru_d | `gru_d.yaml` | N/A | Supervised GRU-D (trains per task) |
| brits | `brits.yaml` | N/A | Supervised BRITS (trains per task) |

## Supported Tasks

33 tasks defined in `src/labels/label_types.json`:

| Task Type | Metrics | Examples |
|-----------|---------|----------|
| Binary | AUROC, AUPRC | Diabetes, Hypertension, BiologicalSex |
| Multiclass | Accuracy, Macro F1 | sleep_time_categories, happiness_categories |
| Ordinal | Spearman R, QWK, MAE | feel_worthwhile1-4, BMI_categories |
| Regression | MSE, MAE, Pearson R | age, BMI_values, WeightKilograms |

## Time Windows

The pipeline supports temporal windowing relative to each user's label date:

| Window | Description |
|--------|-------------|
| `full` | All segments (ignore label date) |
| `before_label` | All segments on or before label date |
| `before_Nw` | Last N weeks before label date |
| `after_Nw` | First N weeks after label date |
| `around_Nw` | +/-N weeks around label date |

Configure via: `--experiment.time_windows "full,before_label,before_10w"`

## Aggregation

Segment-level features/predictions are pooled to user level before classification:

- **mean**: Simple mean over all segments per user (default)
- **cov_weighted_mean**: Coverage-weighted mean where each segment is weighted by
  its coverage fraction (n_valid_hours / hours_per_segment)

## Output

Results CSV at `results/eval/eval_results_{feature_type}.csv` with columns for
task, classifier, time window, sample counts, and all relevant metrics.

## Dependencies

Core: numpy, scikit-learn, pandas, datasets (HuggingFace), jsonargparse

Optional:
- torch, pytorch_lightning (SSL encoder features)
- sktime (MultiRocket features)
- pypots (GRU-D, BRITS supervised models)
- xgboost (XGBoost classifiers)
