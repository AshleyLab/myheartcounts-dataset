# `imputation_training` — PyPOTS imputer training pipeline

OpenMHC's training counterpart to `imputation_evaluation`. Same data
layer, same splits, produces release bundles directly consumable by the
eval CLI's `method.release_dir=` flag.

Supported models: **BRITS, DLinear, TimesNet, FEDformer** — the four
neural imputers benchmarked in the OpenMHC paper.

## Why this package exists

Two reasons:

1. **The public benchmark needs a worked example of how to train an
   imputer.** This package provides it. The Sherlock README at
   `jobs/sherlock/imputation_train/README.md` walks through retraining
   FEDformer end-to-end.

2. **FEDformer reproducibility bug.** PyPOTS' `FourierBlock` stores
   its randomly-shuffled frequency-mode indices on a plain Python
   attribute that is NOT in `state_dict`. Loading a `.pypots`
   checkpoint in a fresh process re-draws the indices against an
   unknown `np.random` state — trained weights then operate on the
   wrong frequency bins (~2–6% NRMSE degradation observed in our
   parity audit).

   This package fixes it via a release-bundle sidecar:
   - At training time, `imputation_training.release.write_release`
     extracts each `FourierBlock.index` and writes them to
     `fourier_modes.json` next to the `.pypots` file.
   - At inference time,
     `openmhc.imputers.pypots.FEDformerImputer._post_load` reads the
     sidecar and assigns the indices back to each module after
     `model.load()`.

   The manifest schema bump (`spec_version=2`) adds an optional
   `fourier_modes` field that points at the sidecar. Spec v1 manifests
   stay loadable so existing paper checkpoints keep working (they just
   exhibit the bug, as before).

## Quick start

```bash
# Install (already in `pip install -e .` if you cloned the repo)
pip install -e ".[pypots,hydra]"

# Tiny smoke run on the xs dataset
mhc-impute-train \
  model=fedformer \
  data.version=xs \
  data.daily_hf_dir=$HOME/.cache/openmhc/data-xs/processed/daily_hf \
  +data.split_file=$HOME/.cache/openmhc/data-xs/splits/sharable_users_seed42_2026_xs.json \
  training.epochs=1 \
  training.batch_size=4 \
  output.release_dir=/tmp/fedformer-smoke

# Use the trained model in the eval pipeline
mhc-impute-eval \
  method=fedformer \
  method.release_dir=/tmp/fedformer-smoke \
  data=xs \
  ...
```

## Pipeline

```
PyPOTSTrainingConfig
  │
  ▼
seeding.seed_everything(seed)        # before model construction!
  │
  ▼
data_export.export_splits_to_h5(...)  # content-addressed H5 cache
  │
  ▼
model_registry.create_model(...)      # BRITS / DLinear / TimesNet / FEDformer
  │
  ▼
model.fit(train_h5, val_h5)           # PyPOTS' own training loop
  │
  ▼
release.write_release(model, ...)     # bundle with manifest + sidecar
  │
  ▼
release_dir/   ← consumable by mhc-impute-eval method.release_dir=...
```

## Public API

```python
from imputation_training import (
    PyPOTSTrainingConfig, ModelConfig, TrainingConfig,
    OutputConfig, H5ExportConfig,
    run_training, seed_everything,
)

cfg = PyPOTSTrainingConfig(
    seed=42,
    model=ModelConfig(model_name="fedformer", n_steps=1440, n_features=19, modes=32),
    training=TrainingConfig(epochs=50, batch_size=64, device="cuda"),
    output=OutputConfig(
        saving_path="/path/to/run",
        release_dir="/path/to/release",
    ),
)
release_dir = run_training(cfg)
```

The Hydra CLI (`mhc-impute-train`) is a thin wrapper around
`run_training(...)`. See `configs/training/` for the YAML schema and
`jobs/sherlock/imputation_train/README.md` for the SLURM walkthrough.

## Code layout

| file | purpose |
|---|---|
| `config.py` | Dataclasses: `PyPOTSTrainingConfig` + 5 sub-configs |
| `seeding.py` | `seed_everything(seed)` — must run before model construction |
| `data_export.py` | Stream `ImputationDataLoader` batches into PyPOTS-compatible HDF5 |
| `normalization.py` | Minimal `ChannelStats` helper for train-time normalization |
| `model_registry.py` | Factory: name → configured PyPOTS model (no `state_dict` yet) |
| `release.py` | Extract FourierBlock indices, package release bundle, write manifest |
| `runner.py` | `run_training(config)` — full orchestrator |
| `hydra/cli.py` | `mhc-impute-train` entry point |

## See also

- `src/imputation_evaluation/README.md` — the eval-side counterpart
- `jobs/sherlock/imputation_train/README.md` — Sherlock SLURM walkthrough
- `src/openmhc/imputers/pypots.py:FEDformerImputer` — inference-side sidecar restoration
- `src/openmhc/imputers/_release.py` — manifest schema (spec v2)
