# OpenMHC

Evaluation API and reference implementations for the **MyHeartCounts Datasets & Benchmarks** (MHC D&B) leaderboard. Wearable sensor data from a real-world cardiovascular cohort, with three benchmark tracks: outcome prediction, imputation, and forecasting.

- **Leaderboard:** https://myheartcounts.stanford.edu/openmhc
- **Submit a result:** [Submit to the leaderboard](#submit-to-the-leaderboard)
- **Paper:** *OpenMHC: Accelerating the Science of Wearable Foundation Models* (NeurIPS 2026)

## Install

```bash
git clone https://github.com/AshleyLab/myheartcounts-dataset.git
cd myheartcounts-dataset
pip install -e .
```

Python ≥ 3.10. Installs the `openmhc` package and its evaluation dependencies.

## Quickstart

Download the tiny version of the dataset (small subset for reviewers and quickstart):

```python
import openmhc
openmhc.download_dataset(version="tiny")
```

Then evaluate a model. Models implement one of three duck-typed protocols — no inheritance required.

### Track 1 — outcome prediction (`Encoder`)

```python
import numpy as np
import openmhc

class MeanPoolEncoder:
    def encode(self, weekly_tensors: np.ndarray) -> np.ndarray:
        # weekly_tensors: (B, 168, 38). Return (B, D) embeddings.
        return weekly_tensors[:, :, :19].mean(axis=1).astype(np.float32)

results = openmhc.evaluate_prediction(MeanPoolEncoder())
print(results.summary())
print("global score (mean AUROC over binary tasks):", results.global_score)
```

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

results = openmhc.evaluate_imputation(MeanImputer())
print(results.summary())
```

### Track 3 — forecasting (`Forecaster`)

```python
import numpy as np
import openmhc

class LastValueForecaster:
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        # history: (n_channels, history_length); returns (n_channels, horizon)
        last = np.nan_to_num(history[:, -1:], nan=0.0)
        return np.tile(last, (1, horizon)).astype(np.float32)

results = openmhc.evaluate_forecasting(LastValueForecaster(), forecasting_length=24)
print(results.summary())
```

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

`to_submission_yaml` returns a paste-ready body matching the textareas in the [submission issue template](../../issues/new?template=submission.yml). For Track 2 imputation, skill scores and per-category subgroup scores are computed locally against the frozen LOCF baseline; for Tracks 1 and 3, those fields are filled in by the maintainers from `raw_metrics` during ingestion (Linear + Seasonal Naive baseline files aren't shipped yet). The HuggingFace Space ingests merged submissions and the public leaderboard rebuilds automatically.

Submissions must follow the standard evaluation protocol — same split file, masking config, and label-validity criterion as the paper. The submission template enforces required fields.

## Repo layout

| Path | What's there |
|---|---|
| `src/openmhc/` | Public API (`evaluate_prediction`, `evaluate_imputation`, `evaluate_forecasting`, `download_dataset`, …) |
| `src/downstream_evaluation/` | Track 1 internals (linear probes, time-window selection, metrics) |
| `src/imputation_evaluation/` | Track 2 internals (masking scenarios, per-channel metrics) |
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
