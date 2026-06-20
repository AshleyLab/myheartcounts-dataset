# OpenMHC

Evaluation API and reference implementations for the **MyHeartCounts Datasets & Benchmarks** (MHC D&B) leaderboard. Wearable sensor data from a real-world cardiovascular cohort, with three benchmark tracks: outcome prediction, imputation, and forecasting.

- **Leaderboard:** https://myheartcounts.stanford.edu/benchmark
- **Submit a result:** [open a submission issue](../../issues/new?template=submission.yml)
- **Paper:** *OpenMHC: Accelerating the Science of Wearable Foundation Models* (NeurIPS 2026)

## Install

```bash
git clone https://github.com/AshleyLab/myheartcounts-dataset.git
cd myheartcounts-dataset

# Install into a dedicated environment (conda for Python, pip for the rest):
conda create -n openmhc python=3.10 -y && conda activate openmhc
pip install -e ".[all]"
```

Python ≥ 3.10. `[all]` pulls in every track (prediction, imputation,
forecasting) plus the Hydra CLIs and W&B logging; use a bare `pip install -e .`
for just Track 1 + the core API. **Install into an isolated environment** — the
evaluation engines use non-unique top-level package names that will collide with
the private `MHC-benchmark` repo if both share one environment.

See [`docs/install.md`](docs/install.md) for the full guide (extras table, venv
alternative, Sherlock cluster setup, verification).

## Quickstart

Download the XS version of the dataset (small subset for reviewers and quickstart):

```python
import openmhc
openmhc.download_dataset(version="xs")
```

Then evaluate a model. Models implement one of three duck-typed protocols — no inheritance required.

### Track 1 — outcome prediction (`Method`)

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

results = openmhc.evaluate_prediction(MeanPoolMethod())
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

### Track 2 — imputation (`Imputer`)

```python
import numpy as np
import openmhc

class MeanImputer:
    def fit(self, data, masks):
        self.means = np.nanmean(data, axis=(0, 2))

    def impute(self, data, observed_mask, target_mask):
        out = data.copy()
        for ch in range(19):
            target = target_mask[:, ch, :] == 1
            out[:, ch, :][target] = self.means[ch]
        return out.astype(np.float32)

results = openmhc.evaluate_imputation(MeanImputer(), version="xs")
print(results.summary())
```

For the paper baselines (BRITS, TimesNet, DLinear, FEDformer, LSM2 daily / weekly /
weekly-sparse), `openmhc.imputers` ships pre-trained-checkpoint wrappers loadable in
one line from a release bundle:

```python
from openmhc.imputers import LSM2Imputer
imp = LSM2Imputer.from_release("path/to/openmhc-lsm2-daily/")
```

See [`docs/neural-imputers.md`](docs/neural-imputers.md) for the full reference
(architecture hyperparameters, paper checkpoint sources, release bundle format,
and the `tools/build_manifest.py` converter for staging your own bundles).

For reproducible runs (W&B logging, SLURM sweeps, manifest-traceable releases),
the `mhc-impute-eval` Hydra CLI composes YAML configs at `configs/imputation/`:

```bash
mhc-impute-eval method=brits method.release_dir=path/to/openmhc-brits-paper/
mhc-impute-eval --multirun method=brits,timesnet,lsm2 masking=all_six
```

See [`src/imputation_evaluation/README.md`](src/imputation_evaluation/README.md#part-15--reproducible-runs-via-mhc-impute-eval) for the full guide (configs, overrides, SLURM dispatch, adding new methods).

### Track 3 — forecasting (`Forecaster`)

```python
import numpy as np
import openmhc

class LastValueForecaster:
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        # history: (n_channels, history_length); returns (n_channels, horizon)
        last = np.nan_to_num(history[:, -1:], nan=0.0)
        return np.tile(last, (1, horizon)).astype(np.float32)

results = openmhc.evaluate_forecasting(LastValueForecaster(), version="xs", forecasting_length=24)
print(results.summary())
```

For reproducible paper-style forecasting runs, use the Hydra CLI and config
family at `configs/forecasting/`:

```bash
mhc-forecast-eval model=seasonal_naive
mhc-forecast-eval --multirun model=seasonal_naive,autoARIMA,autoETS
```

Simurgh (SC) SLURM wrappers live in `jobs/sc-cluster/forecasting_eval/`. The
forecasting implementation supports both device-channel scoring modes: per-task
(default) — phone/watch channels scored separately and combined into
steps/distance scopes by geometric mean, consistent with the imputation track —
and the legacy signal-merge (`--combine-channels`) used for some appendix tables.

See [`src/forecasting_evaluation/README.md`](src/forecasting_evaluation/README.md)
for the full guide (Hydra overrides, release checkpoints, full-data Seasonal
Naive parity checks, offline metrics, raw appendix tables, and SLURM dispatch).

A more complete walkthrough is in [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).

## Submit to the leaderboard

```python
body = results.to_submission_yaml(
    method_name="My Method",
    submitter_team="Stanford CS",
    paper_url="https://arxiv.org/abs/...",
    code_url="https://github.com/...",
)
print(body)
```

`to_submission_yaml` returns a paste-ready body matching the textareas in the [submission issue template](../../issues/new?template=submission.yml). Skill scores, fair skill scores, and average ranks are filled in by the maintainers from `raw_metrics` during ingestion for **all three tracks** (Track 1 vs Linear, Track 2 vs LOCF, Track 3 vs Seasonal Naive). The maintainer-side reducer is a paired per-user geomean of clipped error ratios — MAE for continuous channels, `max(1 − AUC_u, 0.005)` for binary — matching the formula in `forecasting_evaluation/metrics/skill_score_summary.py::compute_skill_from_errors` so the same word means the same thing across tracks. Submitters only paste the absolute per-channel MAE / AUC. The HuggingFace Space ingests merged submissions and the public leaderboard rebuilds automatically.

Submissions must follow the standard evaluation protocol — same split file, masking config, and benchmark task list as the paper. The submission template enforces required fields.

## Repo layout

| Path | What's there |
|---|---|
| `src/openmhc/` | Public API (`evaluate_prediction`, `evaluate_imputation`, `evaluate_forecasting`, `download_dataset`, …) |
| `src/downstream_evaluation/` | Track 1 internals (linear probes, time-window selection, metrics) |
| `src/imputation_evaluation/` | Track 2 internals (masking scenarios, per-channel metrics) |
| `src/imputation_training/` | Track 2 training pipeline — `mhc-impute-train` for BRITS/DLinear/TimesNet/FEDformer. See [`docs/neural-imputers.md`](docs/neural-imputers.md#training-your-own-imputer) |
| `src/forecasting_evaluation/` | Track 3 internals (window cache, point + quantile metrics) |
| `src/labels/` | Label registry + type lookup |
| `data/labels/` | Schema-only registry files (label types, ordinal vocab, validity config) |
| `notebooks/quickstart.ipynb` | End-to-end example |
| `.github/ISSUE_TEMPLATE/submission.yml` | Leaderboard submission form |

The actual participant data lives separately — see [DATASET.md](DATASET.md) for download instructions and split conventions.

## Citation

```bibtex
@inproceedings{openmhc2026,
  title     = {OpenMHC: Accelerating the Science of Wearable Foundation Models},
  author    = {MyHeartCounts Team},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2026}
}
```

## License

Code: MIT. Dataset: governed by a separate Data Use Agreement — see [DATASET.md](DATASET.md).
