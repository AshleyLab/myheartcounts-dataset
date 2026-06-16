# Downstream Prediction Evaluation

Evaluate any wearable-sensor model on the 32 MyHeart Counts health-prediction tasks. One contract for every model — bundled baseline and external submission alike: `fit(data, labels, task_type)` / `predict(data)` on per-participant arrays.

- [How it works](#how-it-works)
- [Reproduce the paper results](#reproduce-the-paper-results)
- [Reproducible runs via the CLI](#reproducible-runs-via-the-cli-mhc-downstream-eval)
- [Evaluate your own model](#evaluate-your-own-model)

## How it works

One engine evaluates every model on identical cohorts:

```
openmhc.evaluate_prediction(model, tasks)
  └─ per task:
       TaskDataProvider   who & what — cohort user_ids, labels, eligible days
       DataLoader         the bytes — raw segments, one dataset read per run
       model.fit(train data, labels, task_type)
       model.predict(test data)  →  scored on the held-out cohort
```

**Data layer** (`src/downstream_evaluation/data/`):

- **`provider.py` — who and what.** `TaskDataProvider` reads the labels lookup (one parquet per granularity) and derives, per `(task, split)`: the cohort, their labels, and the dates of each user's eligible days.
- **`loader.py` — the bytes.** `DataLoader` reads `daily_hourly_hf` once per run (lazily, on first access), indexes it by `(user_id, date)`, and serves every access pattern: `bind()` (a task cohort's eligible segments), `segment_store()` (the whole store, for global-fit models), `user_days()` (date-ascending days, for timeline builders), `as_daily_rows()` (raw-form rows, for window-index consumers).
- **`splits.py`** reads the frozen train/validation/test user split.

Quality gating happens once, upstream, when `daily_hourly_hf` is built; eligibility is whatever the lookup names. The loader selects those `(user, date)` rows and never re-filters — no model decides its own cohort. (`mae`/`xgboost` read at minute resolution through the loader's `participant_minute` path over the `daily_hf` store — the same single-read discipline, no separate extraction.)

**Scoring.** For each task the benchmark fits your model on the train cohort and scores its predictions on the held-out test cohort. Encoder-style models all run the *same* probe (`openmhc.LinearProbe`: PCA-50 + a linear head) inside `fit`/`predict`, so the comparison isolates representation quality; end-to-end models own their head. Primary metric per task type:

| Task type  | Metric     | Example tasks                          |
|------------|------------|----------------------------------------|
| Binary     | AUPRC      | Diabetes, Hypertension, BiologicalSex  |
| Ordinal    | Spearman ρ | BMI_categories, feel_worthwhile1-4     |
| Regression | Pearson r  | age, BMI_values, WeightKilograms       |

## Reproduce the paper results

Three steps: data, baselines, paper pipeline.

**1. Get the dataset** (see `DATASET.md`) and point `MHC_DATA_DIR` at its root.

**2. Run the eight baselines.** One driver for every method; predictions are the paper pipeline's input, so set `PREDICTIONS_DIR`:

```bash
for M in linear multirocket gru_d xgboost mae toto chronos2 wbm; do
  METHOD=$M MHC_DATA_DIR=path/to/mhc-data \
  PREDICTIONS_DIR=results/eval/final/predictions \
  OUT_CSV=results/eval/final/eval_$M.csv \
  sbatch jobs/imperial/slurm/run_eval.slurm        # add --gres=gpu:1 for toto/chronos2/wbm/mae/gru_d
done
```

| METHOD          | Model                                          | Notes                          |
|-----------------|------------------------------------------------|--------------------------------|
| linear          | per-channel mean/std (+ demographics)          | CPU                            |
| multirocket     | random convolutional kernels                   | CPU                            |
| gru_d           | end-to-end GRU-D (trains per run)              | GPU; see reproducibility below |
| xgboost         | gradient-boosted trees on minute-level features| CPU; needs `daily_hf`          |
| mae             | masked-autoencoder embeddings                  | GPU on cache miss; `daily_hf`  |
| toto / chronos2 | time-series foundation-model embeddings        | GPU on cache miss              |
| wbm             | self-supervised weekly encoder + Linear hybrid | GPU on cache miss              |

**3. Run the paper pipeline** (bootstrap CIs, skill/rank/fairness aggregates) — one config drives every phase:

```bash
PYTHONPATH=src python scripts/downstream_paper_results/run_paper_pipeline.py \
    --config configs/paper/downstream_paper.yaml
```

`configs/paper/downstream_paper.yaml` records the methods, bootstrap count/seed, baseline, and the fairness knobs — the single provenance of the published numbers. Re-aggregate without re-bootstrapping via `--skip-phase1`.

## Reproducible runs via the CLI (`mhc-downstream-eval`)

The bundled baselines also run from a composable [Hydra](https://hydra.cc) CLI — the prediction-track twin of `mhc-impute-eval` / `mhc-forecast-eval`. It builds the model and calls the same `openmhc.evaluate_prediction` engine as the snippets above; reach for it (over the `METHOD=… scripts/run_eval.py` env-var driver) when you want config provenance, parameter sweeps, or cluster dispatch — each run snapshots its fully-resolved config next to the results.

```bash
pip install -e ".[hydra]"          # or, from the published package: pip install "openmhc[hydra]"
mhc-downstream-eval method=xgboost
# no install needed for a one-off:
PYTHONPATH=src python -m downstream_evaluation.hydra.cli method=xgboost
```

**Configs.** Composable YAML under [`configs/downstream/`](../../configs/downstream/), one group per axis; override any field on the command line:

```
configs/downstream/
  eval.yaml               # defaults list + run-dir layout
  data/default.yaml       # data_dir (null -> MHC_DATA_DIR)
  method/<name>.yaml      # one per bundled model (8): type + build-on-miss knobs
  evaluation/default.yaml # tasks: all | [age, Diabetes, ...]
  output/default.yaml     # results_dir, predictions_dir
```

**Usage.**

```bash
mhc-downstream-eval method=linear                       # one method, all 32 tasks
mhc-downstream-eval method=xgboost evaluation.tasks=[age,Diabetes] \
    data.data_dir=/path/to/mhc-data output.predictions_dir=results/eval/predictions
mhc-downstream-eval --multirun method=linear,mae,xgboost   # sweep (one run dir each)
```

**Output.** Each run lands in `${output.results_dir}/<timestamp>_<method>/`:

```
eval.csv               # long-format per-(task, metric) results — same schema as results/eval/final/
resolved_config.yaml   # the fully-resolved config (provenance)
.hydra/                # Hydra's own config + overrides snapshot
cli.log                # run log
```

**Cluster dispatch (PBS).** Pick the queue by whether the method needs a GPU. CPU-only methods (`linear`, `multirocket`, `xgboost`) use `run_eval.pbs`:

```bash
qsub -v METHOD=xgboost,MHC_DATA_DIR=$HOME/mhc-data jobs/imperial/pbs/run_eval.pbs
```

GPU methods use `run_eval_gpu.pbs` (`:ngpus=1`): `gru_d` (trains every run), and `toto`/`chronos2`/`wbm`/`mae` for embedding extraction on a cache miss (a warm cache runs them CPU-only). CPU jobs request `mem=64gb` — the hourly loader materializes the `daily_hourly_hf` cohort in RAM (~32 GB peak), so they do not fit an interactive login session.

**Add a method.** Register a builder in [`hydra/registry.py`](hydra/registry.py) and drop a `configs/downstream/method/<name>.yaml` with `type: <name>`. The builder takes `(method_cfg, data_cfg)` and returns `(model, None)` — the same model object you would otherwise hand to `openmhc.evaluate_prediction`.

## Evaluate your own model

Implement `fit` / `predict` and hand the model to the benchmark — no base class, no config files. The one contract supports two styles; pick by **who owns the classification head**:

- **Encoder-style** — your model produces a representation; the benchmark's uniform head (`openmhc.LinearProbe`) turns it into predictions. Your score reflects the *representation* and is directly comparable with the paper's encoder rows (mae, toto, chronos2, multirocket).
- **End-to-end** — your model owns its head and returns predictions directly, scored as-is. Comparable with the paper's end-to-end rows (gru_d, xgboost).

**Encoder-style** (uniform probe inside `fit`/`predict`):

```python
import numpy as np
import openmhc
from openmhc import DataSpec

class MyEncoderMethod:
    data_spec = DataSpec("hourly", "day")    # your input shape — see "Input shapes" below

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

results = openmhc.evaluate_prediction(MyEncoderMethod(), tasks="all", data_dir="path/to/mhc-data")

print(results.summary())             # wide table: one row per task, one column per metric
results.to_csv("my_results.csv")     # full long-format results
```

**End-to-end** (your own head — here a random forest, routed by task type):

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from openmhc import DataSpec

class MyEndToEndMethod:
    data_spec = DataSpec("hourly", "day")

    def _features(self, data):
        return np.stack([np.nan_to_num(x).reshape(-1, 38).mean(0) for x in data])

    def fit(self, data, labels, task_type):
        cls = RandomForestRegressor if task_type == "regression" else RandomForestClassifier
        self._model = cls(n_estimators=300, random_state=42).fit(self._features(data), labels)
        self._task_type = task_type

    def predict(self, data):
        X = self._features(data)
        if self._task_type == "binary":
            return self._model.predict_proba(X)[:, 1]   # scores, not hard labels (AUPRC)
        return self._model.predict(X)                    # ordinal/multiclass: levels; regression: values
```

**Input shapes.** Declare what each participant's data looks like with `data_spec`:

| `data_spec` | each participant |
|---|---|
| `DataSpec("hourly", "day")` | `(n_days, 24, 38)` |
| `DataSpec("hourly", "series", N)` | `(N, 38)` — one continuous window |
| `DataSpec("minute", "day")` | `(n_days, 1440, 38)` |

Channels 0-18 are sensor values (NaN at missing), 19-37 the missingness mask. Iterate `data` for one participant at a time — `minute` (and other large) specs stream, so don't index `data[i]` or stack the whole cohort.

**Memory.** Why iterate instead of stacking? For `hourly` specs the whole cohort is small (a few GB), so `data` is just a list. For `minute` (and other large) specs the full cohort is hundreds of GB — far past RAM — so the benchmark *streams* it: `data` is a `CohortView` that hands you one participant at a time and never holds the whole cohort at once. Peak memory then stays at roughly one participant (plus whatever you accumulate), **independent of cohort size** — as long as you iterate and don't stack the raw `data`.

**The contract.** You implement two methods.

`fit(data, labels, task_type)` — train on the cohort:

- `data` — iterate it, one participant's array at a time (a list; a streamed `CohortView` for large specs — see *Input shapes*).
- `labels` — one per participant, aligned with the cohort.
- `task_type` — `"binary"`, `"multiclass"`, `"ordinal"`, or `"regression"`.

`predict(data)` — return one prediction per participant, in `data` order. What to return depends on the task:

| `task_type` | return per participant |
|---|---|
| `binary` | score / probability of the positive class — a ranking, **not** 0/1 labels (AUPRC needs it) |
| `multiclass` / `ordinal` | the class level |
| `regression` | the value |

The benchmark owns the cohort and eligibility — who's included, and which of each participant's dates count (full history by default) — so your model only ever sees eligible data and can't get the cohort wrong. Encoder vs end-to-end may train, pretrain, or be training-free; the style only decides which paper rows you're comparable with.

Run it with `openmhc.evaluate_prediction(model)` → a `PredictionResults` (`.summary()`, `.to_csv()`, `.to_json()`, `.to_dataframe()`). `tasks="all"` runs all 32 tasks (`openmhc.list_tasks()`); `data_dir` defaults to `$MHC_DATA_DIR`.

## Requirements

`numpy`, `scikit-learn`, `pandas`, `datasets`. The foundation-model baselines (`toto`, `chronos2`, `wbm`, `mae`) additionally need `torch` and a GPU for embedding extraction (cached after the first run).
