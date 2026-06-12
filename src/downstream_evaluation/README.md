# Downstream Prediction Evaluation

Evaluate any wearable-sensor model on the MyHeart Counts health-prediction tasks.
One contract for every model: `fit(data, labels, task_type)` / `predict(data)` on
per-participant arrays. An encoder-style model runs the benchmark's **uniform linear
probe** (`openmhc.LinearProbe`) on its embeddings inside `fit` / `predict` — so its
results reflect the *representation*, not a choice of classifier — while an
end-to-end model owns its head.

## Evaluate your own model

Implement `fit` / `predict` and hand the model to the benchmark.
No base class, no config files:

```python
import numpy as np
import openmhc

class MyMethod:
    input_granularity = "daily"          # the benchmark hands you daily segments

    def _encode(self, data: np.ndarray) -> np.ndarray:
        # data: (n_days, 24, 38) — one participant's eligible days.
        #   channels 0-18 = raw sensor values (NaN at missing positions)
        #   channels 19-37 = missingness mask (1 = missing, 0 = observed)
        # Normalize however your model needs; return any vector of length >= 50.
        x = np.nan_to_num(data).reshape(-1, 38)
        return np.concatenate([x.mean(0), x.std(0)])      # -> (76,)

    def fit(self, data, labels, task_type):
        emb = np.stack([self._encode(x) for x in data])
        self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

    def predict(self, data):
        return self._probe.predict(np.stack([self._encode(x) for x in data]))

results = openmhc.evaluate_prediction(MyMethod(), tasks="all", data_dir="path/to/mhc-data")

print(results.summary())             # wide table: one row per task, one column per metric
results.to_csv("my_results.csv")     # full long-format results
```

`evaluate_prediction(model, tasks="all", data_dir=None, seed=42)` returns a
`PredictionResults` with `.summary()`, `.to_csv()`, `.to_json()`, and `.to_dataframe()`.
Set `data_dir` to the dataset root (or the `MHC_DATA_DIR` env var). List the tasks with
`openmhc.list_tasks()`.

## How scoring works

For each task the benchmark:

1. selects each eligible participant's data (cohort + time window are handled for you),
2. calls your `fit` on the train cohort's data + labels,
3. scores your `predict` on the held-out test cohort.

Encoder-style models all run the *same* probe (`openmhc.LinearProbe`: PCA-50 + a
linear head), so the comparison isolates representation quality. Primary metric per
task type:

| Task type  | Metric     | Example tasks                          |
|------------|------------|----------------------------------------|
| Binary     | AUPRC      | Diabetes, Hypertension, BiologicalSex  |
| Ordinal    | Spearman ρ | BMI_categories, feel_worthwhile1-4     |
| Regression | Pearson r  | age, BMI_values, WeightKilograms       |

## Data layer

Three small modules in `src/downstream_evaluation/data/` feed every evaluation:

- **`provider.py` — who and what.** `TaskDataProvider` reads the labels lookup (one
  parquet per granularity, built by `scripts/labels/build_labels_lookup.py`) and
  derives, per `(task, split)`: the cohort (`user_ids`), their `labels`, and the
  `dates` of each user's eligible days (the days passing the inclusion criteria).
- **`loader.py` — the bytes.** `DataLoader` reads `daily_hourly_hf` **once per run**,
  indexes it by `(user_id, date)`, and materializes per-participant segments:
  `bind(task_data)` fills a task cohort's eligible segments, and `segment_store()`
  exposes the whole store for global-fit models (GRU-D, MultiRocket).
- **`splits.py`** reads the frozen train/validation/test user split.

The loader is deliberately dumb about data quality: the wear-time gate is applied
once upstream when `daily_hourly_hf` is built, and eligibility is whatever the
lookup names — the loader selects those `(user, date)` rows and never re-filters.

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
