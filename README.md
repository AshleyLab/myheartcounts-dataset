# OpenMHC

[![Project Page](https://img.shields.io/badge/Project%20Page-MyHeartCounts-1f883d?logo=googlechrome&style=flat-square)](https://myheartcounts.stanford.edu/)
[![Benchmark](https://img.shields.io/badge/HuggingFace-Benchmark-ffd21e?style=flat-square&logo=huggingface)](https://myheartcounts-openmhc.hf.space)
[![Models](https://img.shields.io/badge/HuggingFace-Models-ffd21e?style=flat-square&logo=huggingface)](https://huggingface.co/MyHeartCounts/models)
[![Dataset](https://img.shields.io/badge/Dataset-Coming%20Soon-6c757d?style=flat-square)](#)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b?style=flat-square)](#)

This is the official public repository and package of OpenMHC, an Apple HealthKit based mobile and wearable dataset and evaluation suite with over 60M hours of minute-level data from around 12k participants across multiple countries. This repository includes the evaluation harnesses to recreate the public leaderboards, contains a public API to run your own methods on the benchmark and create results files that can be submitted (see below for more details). The repo also comes with reference implementations of models presented in the OpenMHC paper, including reimplementations/adaptations of Google's LSM2 and Apple's WBM. 
The research-grade codebased can be found here https://github.com/NarayanSchuetz/OpenMHC (particularly relevant for training infra, until we ported that properly).

What can this repository and the OpenMHC dataset be useful for?
- Experiment on the, to-date, largest fully public mobile/wearable health dataset. This space is vastly underexplored in the AI/ML world and will allow for many valuable contributions that could affect millions of users one day.
- Leverage our pre-trained wearable foundation models on your own datasets (this could be from research or your own Apple HealthKit exports - we are working on adaptors for the latter but Claude/Codex can likely help prior to that), or pre-train you models on our large-scale dataset.
- Evaluate your own new method/model on our benchmark tasks, spanning dense downstream prediction across >30 tasks, 24h wearable data forecasting, a single day/multi-day minute level imputation and submit them to be displayed on the official leaderboard
- Augment your own datasets, in mobile/wearable health studies we are often limite to small disease cohorts, OpenMHC can give you access to larger numbers of subjects to compare to.


- **Dataset:** see [DATASET.md](DATASET.md)

## Release Plan

- [ ] Release Full OpenMHC dataset (Estimated: August-December)
- [ ] Release Apple HealthKit export adaptor, so people can directly run our models on their data (Estimated: July-September)
- [ ] Release cleaned-up training infrastructure here (Estimated: July-August)
- [x] Release model checkpoints on Hugging Face
- [x] Release benchmark Hugging Face Space
- [x] Release evaluation code
- [x] Release OpenMHC-XS dataset
- [x] Release paper on arXiv

## Install

```bash
git clone https://github.com/AshleyLab/myheartcounts-dataset.git
cd myheartcounts-dataset

conda create -n openmhc python=3.10 -y
conda activate openmhc
pip install -e ".[all]"
```

OpenMHC requires Python 3.10 or newer. The base install supports the public API
and core evaluation code; `.[all]` adds optional model wrappers, Hugging Face
bundle downloads, Hydra CLIs, and W&B logging. Use `pip install -e ".[all,dev]"`
for contributor tooling.

Install OpenMHC in an isolated environment. Some evaluation engines use generic
top-level package names that can collide with private/internal benchmark repos if
both are installed into the same Python environment.

See [docs/install.md](docs/install.md) for the full install guide, optional
extras, virtualenv setup, Sherlock setup, and verification commands.

## Dataset Setup

The dataset is hosted separately from this code repository. The `xs` release is
available for quickstarts and smoke tests; the `full` release uses the same
layout and API contract once available. Download each version into its own root
directory and pass the version explicitly when evaluating:

```python
import openmhc

openmhc.download_dataset(version="xs", dest="~/.cache/openmhc/data-xs")
```

Then either pass `data_dir=` to each evaluation call or set `MHC_DATA_DIR`:

```bash
export MHC_DATA_DIR=~/.cache/openmhc/data-xs
```

Every dataset root must contain `dataset_version.json`. `download_dataset`
writes it automatically, and the evaluation API cross-checks it against the
`version=` argument so an `xs` run cannot accidentally score against a full-data
root, or vice versa.

See [DATASET.md](DATASET.md) for Dataverse details, manual setup, directory
layout, and Data Use Agreement terms.

## Quickstart

Models implement small protocols; no inheritance is required. All evaluation
entry points require `version="xs"` or `version="full"` and resolve large data
payloads from `data_dir=` or `MHC_DATA_DIR`.

### Outcome Prediction

Implement a `Method` with `predict(data)` and, for trainable methods,
`fit(data, labels, task_type)`. Each `data` item is one participant's eligible
daily segments with shape `(n_days, 24, 38)`: channels 0-18 are raw sensor
values with NaN at missing positions, and channels 19-37 are missingness masks.

```python
import numpy as np
import openmhc

class MeanPoolMethod:
    def _encode(self, data: np.ndarray) -> np.ndarray:
        # data: (n_days, 24, 38) — one participant's eligible days
        # (channels 0-18 raw values with NaN at missing, 19-37 the mask).
        x = np.nan_to_num(data).reshape(-1, 38)
        return np.concatenate([x.mean(0), x.std(0)])

    def fit(self, data, labels, task_type):
        emb = np.stack([self._encode(x) for x in data])
        self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

    def predict(self, data):
        return self._probe.predict(np.stack([self._encode(x) for x in data]))

results = openmhc.evaluate_prediction(MeanPoolMethod(), version="xs")
print(results.summary())
```

`predict` is the only required method; `fit` is optional — a zero-shot / pretrained
model omits it and is scored as-is.

For reproducible runs (config provenance, sweeps, cluster dispatch), the
`mhc-downstream-eval` Hydra CLI runs the bundled baselines from composable configs
at `configs/downstream/`:

```bash
mhc-downstream-eval method=xgboost
mhc-downstream-eval --multirun method=linear,mae,xgboost
```

The full Track-1 guide (data contract, baselines, CLI, paper reproduction) is in
[`src/downstream_evaluation/README.md`](src/downstream_evaluation/README.md).

### Imputation

Implement `impute(data, observed_mask, target_mask) -> imputed_data`, where
`data` has shape `(N, 19, T)`, `T=1440` for daily evaluation by default, and
artificially masked cells are marked by `target_mask == 1`.

```python
import numpy as np
import openmhc


class ZeroImputer:
    def impute(self, data, observed_mask, target_mask):
        out = data.copy()
        out[target_mask == 1] = 0.0
        return out.astype(np.float32, copy=False)


results = openmhc.evaluate_imputation(ZeroImputer(), version="xs")
print(results.summary())
```

The harness only calls `impute`; custom methods should load checkpoints,
compute training statistics, or build per-user state before evaluation, usually
in `__init__`. Optional keyword-only metadata is forwarded only when declared in
the imputer signature: `sample_indices`, `user_ids`, `dates`, and `day_offsets`
for multi-day windows. Use `n_days=7` for weekly imputers, `max_samples=` for
smoke runs, and `output_dir=` / `baseline_errors=` when you need
`per_user_errors.parquet` and paired skill scores.

Reference methods are available from `openmhc.imputers`:

```python
from openmhc.imputers import LOCFImputer, MeanImputer

mean = MeanImputer(version="xs")
locf = LOCFImputer(version="xs")
```

Trained neural imputer wrappers can load local or Hugging Face release bundles:

```python
from openmhc.imputers import LSM2Imputer

imputer = LSM2Imputer.from_release("path/to/openmhc-lsm2-daily/")
```

Use `mhc-impute-eval` for reproducible config-driven runs:

```bash
mhc-impute-eval method=mean data=xs
mhc-impute-eval --multirun method=locf,mean,temporal_mean masking=all_six
```

See [src/imputation_evaluation/README.md](src/imputation_evaluation/README.md)
and [docs/neural-imputers.md](docs/neural-imputers.md) for method contracts,
release bundle format, metrics, and SLURM workflows.

### Forecasting

Implement `predict(history, horizon) -> forecast`, where `history` has shape
`(n_channels, history_length)` and the return value has shape
`(n_channels, horizon)`.

```python
import numpy as np
import openmhc


class LastValueForecaster:
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        last = np.nan_to_num(history[:, -1:], nan=0.0)
        return np.tile(last, (1, horizon)).astype(np.float32)


results = openmhc.evaluate_forecasting(
    LastValueForecaster(),
    version="xs",
    forecasting_length=24,
    max_samples=10,
)
print(results.summary())
```

Reference forecasting wrappers live in `openmhc.forecasters`, and
`mhc-forecast-eval` provides the config-driven evaluation path:

```bash
MHC_DATA_DIR=~/.cache/openmhc/data-xs mhc-forecast-eval model=seasonal_naive
mhc-forecast-eval --multirun model=seasonal_naive,autoARIMA,autoETS
```

See
[src/forecasting_evaluation/README.md](src/forecasting_evaluation/README.md)
and [docs/neural-forecasters.md](docs/neural-forecasters.md) for forecasting
data contracts, model configs, offline metrics, release checkpoints, and
cluster dispatch.

A notebook walkthrough is available at
[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## Public API

The main package exports:

| API | Purpose |
|---|---|
| `openmhc.evaluate_prediction` | Evaluate a `Method` on outcome-prediction tasks |
| `openmhc.evaluate_imputation` | Evaluate an `Imputer` across masking scenarios |
| `openmhc.evaluate_forecasting` | Evaluate a `Forecaster` on hourly forecasting windows |
| `openmhc.download_dataset` | Download a public dataset release from Dataverse |
| `openmhc.data_dir` | Resolve an explicit dataset root or `MHC_DATA_DIR` |
| `openmhc.write_dataset_marker` | Backfill `dataset_version.json` for manual dataset roots |
| `openmhc.iter_train_data` / `iter_split_data` | Iterate sensor data for custom methods |
| `openmhc.list_tasks` | List outcome-prediction labels |
| `openmhc.list_masking_scenarios` | List imputation masking scenarios |
| `openmhc.SENSOR_CHANNELS` | Ordered sensor-channel names |

Result objects provide `summary()`, `to_dataframe()`, `to_csv()`, `to_json()`,
and `to_submission_yaml(...)`.

## Submit to the Leaderboard

Submissions are pull requests on the Hugging Face leaderboard dataset
[`MyHeartCounts/OpenMHC-leaderboard-data`](https://huggingface.co/datasets/MyHeartCounts/OpenMHC-leaderboard-data).
A submission adds two files under the track's subdirectory:

- `<track>/<method>.parquet` — the per-user substrate from your evaluation run
  (Track 2: `per_user_errors.parquet`, written when you pass `output_dir=` and
  `method_name="<method>"` to `evaluate_imputation`). The `method_name` sets the
  parquet's `method` column and **must match the `<method>` filename stem** — it
  defaults to `"custom"`, and the leaderboard groups submissions by that column,
  so an unset name collides with every other default submission.
- `<track>/<method>.meta.json` — the display sidecar
  (`display_name`, `type`, `submitter`, `subtrack`).

Track 2 (imputation) is live today. The Track 1 and Track 3 subdirectory names
and substrate formats are still being finalized — `to_submission_yaml` flags
this in the rendered packet; confirm against `tools/leaderboard_docs/` before
submitting to those tracks.

`to_submission_yaml` renders the `meta.json` block plus the PR file checklist so
you don't hand-write the sidecar:

```python
packet = results.to_submission_yaml(
    method_name="My Method",
    submitter_team="Stanford CS",
    code_url="https://github.com/...",
    paper_url="https://arxiv.org/abs/...",
)
print(packet)
```

Note the two distinct `method_name` arguments: the one above is the **display
label** rendered in `meta.json` and can be free-form (`"My Method"`). The one you
pass to `evaluate_imputation` sets the parquet's `method` column and **must equal
the `<method>` filename stem** the leaderboard groups by.

Lay the two files out under the track subdirectory and open the PR with the
Hugging Face Hub client (`pip install -e ".[hf]"`):

```python
from huggingface_hub import HfApi

# my_submission/imputation/<method>.parquet
# my_submission/imputation/<method>.meta.json
HfApi().upload_folder(
    repo_id="MyHeartCounts/OpenMHC-leaderboard-data",
    repo_type="dataset",
    folder_path="my_submission",
    create_pr=True,
    commit_message="Add <method> to the imputation leaderboard",
)
```

The call returns a PR URL. (You can also drag the files into the dataset's
"Community → New pull request" page on the Hub.)

Public submissions must use the standard evaluation protocol for the selected
track, including the canonical dataset version, split file, masking or
forecasting configuration, and label-validity criterion.

Maintainers compute leaderboard-level skill scores, fair skill scores, and
average ranks from the submitted substrate during ingestion. Track 1 is scored
against the linear-probe baseline, Track 2 against LOCF, and Track 3 against
Seasonal Naive. See
[`tools/leaderboard_docs/imputation/SCHEMA.md`](tools/leaderboard_docs/imputation/SCHEMA.md)
for the Track 2 per-method substrate columns and dtypes, and
`tools/upload_leaderboard_substrate.py` for the maintainer upload path.

## Repo Layout

| Path | What's there |
|---|---|
| `src/openmhc/` | Public API, result containers, protocol definitions, dataset helpers |
| `src/openmhc/imputers/` | Reference imputation methods and release-bundle wrappers |
| `src/openmhc/forecasters/` | Reference forecasting wrappers and release-bundle loaders |
| `src/downstream_evaluation/` | Outcome-prediction internals: linear probes, windows, metrics |
| `src/imputation_evaluation/` | Imputation internals: masks, evaluation loop, metrics |
| `src/imputation_training/` | Imputation training pipeline and `mhc-impute-train` |
| `src/forecasting_evaluation/` | Forecasting internals: sample index, metrics, writers |
| `src/forecasting_training/` | Forecasting training pipeline and `mhc-forecast-train` |
| `src/labels/`, `src/context/`, `src/devices/` | Metadata APIs for labels, context variables, and device resolution |
| `configs/` | Hydra configs for evaluation, training, sweeps, and output layout |
| `jobs/` | SLURM wrappers for Sherlock and SC cluster runs |
| `tools/` and `scripts/` | Release, leaderboard, parity, and paper-result utilities |
| `data/labels/` | Bundled schema-only label metadata |
| `data/imputation/masks/` | Bundled precomputed imputation masks for reproducible scoring |
| `notebooks/quickstart.ipynb` | End-to-end example notebook |
| `tools/leaderboard_docs/` | Docs mirrored into the HF leaderboard dataset repo |

The participant data itself is not tracked in this repository. See
[DATASET.md](DATASET.md) for download instructions and expected layout.

## Citation

Citation information will be added when the public manuscript and full dataset
release are available.

## License

Code: MIT. Dataset: governed by a separate Data Use Agreement (see
[DATASET.md](DATASET.md)), Models OpenRAIL;
