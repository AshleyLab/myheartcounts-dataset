# Downstream Prediction Evaluation

Evaluate any wearable-sensor model on the MyHeart Counts health-prediction tasks.
You declare the input your model wants and implement one method — an **`Encoder`**
(return a representation; the benchmark scores it with a *uniform* linear probe, so
results reflect your representation, not your classifier) or a **`Predictor`** (bring
your own head; the benchmark scores your predictions directly). No base class, no config.

## Evaluate your own model

**1. Declare `input`** — what you receive per participant. Channels 0-18 are sensor
values (NaN at missing positions), 19-37 the missingness mask (1 = missing, 0 = observed):

| `input =` | per participant |
|---|---|
| `openmhc.Raw("hourly")` / `Raw("minute")` | eligible raw days at that resolution, `(n_days, 24\|1440, 38)` — you window/featurize it yourself |
| `openmhc.Window(hours=H)` | one anchored history window, `(1, H, 38)` |

The cohort and the allowed time window come from the dataset (handled for you, no leakage);
you only choose the shape. **Either contract below can use any input — they're independent:**
the input sets the *shape you receive*; the contract (`Encoder` vs `Predictor`) sets *what you
return*. So an encoder on minute data (`input = openmhc.Raw("minute")`) is just as valid.

**2a. `Encoder`** — return a representation of length ≥ 50; the benchmark fits PCA-50 + a
uniform linear probe on it:

```python
import numpy as np
import openmhc

class MyEncoder:
    input = openmhc.Raw("hourly")                    # (n_days, 24, 38) per participant
    def encode(self, data: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(data).reshape(-1, 38)
        return np.concatenate([x.mean(0), x.std(0)])      # any vector, length >= 50

results = openmhc.evaluate_prediction(MyEncoder(), data_dir="path/to/mhc-data")
print(results.summary())
results.to_csv("my_results.csv")
```

A time-series foundation model is the same, with a context window:
`input = openmhc.Window(hours=2048)`.

**2b. `Predictor`** — bring your own classifier head; the benchmark scores your predictions
directly:

```python
class MyModel:
    input = openmhc.Raw("minute")                    # list of (n_days, 1440, 38); you featurize it
    def fit(self, data, labels):
        X = np.stack([featurize(d) for d in data])
        self.clf = SomeClassifier().fit(X, labels)
    def predict(self, data):
        return self.clf.predict_proba(np.stack([featurize(d) for d in data]))[:, 1]

results = openmhc.evaluate_prediction(MyModel(), data_dir="path/to/mhc-data")
```

`evaluate_prediction(model, tasks="all", data_dir=None, seed=42)` returns a
`PredictionResults` with `.summary()`, `.to_csv()`, `.to_json()`, `.to_dataframe()`,
and `.global_score` (mean AUPRC over the binary tasks). Set `data_dir` (or the
`MHC_DATA_DIR` env var). List the tasks with `openmhc.list_tasks()`.

## How scoring works

For each task the benchmark selects each eligible participant's data (cohort + time window
handled for you), then:

- an **`Encoder`** is called once per participant; the benchmark fits PCA-50 + a linear probe
  on the train split and scores the test split — every encoder goes through the *same* probe,
  so the comparison isolates representation quality;
- a **`Predictor`** is `fit` on the train cohort and `predict`s the test cohort; its predictions
  are scored directly.

Primary metric per task type:

| Task type  | Metric     | Example tasks                          |
|------------|------------|----------------------------------------|
| Binary     | AUPRC      | Diabetes, Hypertension, BiologicalSex  |
| Ordinal    | Spearman ρ | BMI_categories, feel_worthwhile1-4     |
| Regression | Pearson r  | age, BMI_values, WeightKilograms       |

## Run the bundled baselines

The shipped baselines run through the *same* `evaluate_prediction` call, selected by
`METHOD`. One driver for every method:

```bash
# CPU methods
METHOD=linear      MHC_DATA_DIR=path/to/mhc-data sbatch              jobs/imperial/slurm/run_eval.slurm
METHOD=multirocket MHC_DATA_DIR=path/to/mhc-data sbatch              jobs/imperial/slurm/run_eval.slurm
METHOD=gru_d       MHC_DATA_DIR=path/to/mhc-data sbatch              jobs/imperial/slurm/run_eval.slurm

# GPU methods (the first run extracts embeddings from raw)
METHOD=toto        MHC_DATA_DIR=path/to/mhc-data sbatch --gres=gpu:1 jobs/imperial/slurm/run_eval.slurm
METHOD=chronos2    MHC_DATA_DIR=path/to/mhc-data sbatch --gres=gpu:1 jobs/imperial/slurm/run_eval.slurm
METHOD=wbm         MHC_DATA_DIR=path/to/mhc-data sbatch --gres=gpu:1 jobs/imperial/slurm/run_eval.slurm
```

| METHOD          | Model                                          |
|-----------------|------------------------------------------------|
| linear          | per-channel mean/std (+ demographics)          |
| multirocket     | random convolutional kernels                   |
| toto / chronos2 | time-series foundation-model embeddings        |
| wbm             | self-supervised wearable encoder (hybrid)      |
| gru_d           | end-to-end GRU-D (trains a model per run)      |

Results are written to `eval_<METHOD>.csv` (override with `OUT_CSV=`). `gru_d` owns its
own classifier and is scored end-to-end; the rest are encoders scored with the probe above.

## Requirements

`numpy`, `scikit-learn`, `pandas`, `datasets`. The foundation-model baselines
(`toto`, `chronos2`, `wbm`) additionally need `torch` and a GPU.
