# Installation

How to install `openmhc` and its evaluation dependencies in a clean, isolated
environment. For getting the dataset itself, see
[`manual-dataset-setup.md`](manual-dataset-setup.md).

## Prerequisites

- **Python ≥ 3.10**
- An **isolated environment** dedicated to `openmhc` (see the warning below).

> [!WARNING]
> **Install `openmhc` into its own environment.** The evaluation engines ship
> top-level packages — `forecasting_evaluation`, `imputation_evaluation`,
> `downstream_evaluation`, `forecasting_training`, `imputation_training`,
> `eval_hydra`, `labels`, `data`, `context`, `devices`, `utils` — whose names
> are **not unique** to this repo. The private `MHC-benchmark` repo defines the
> same names. If both
> are installed (editable) into one environment, whichever sits earlier on
> `sys.path` silently shadows the other, and imports resolve to the wrong copy
> (e.g. `ModuleNotFoundError: No module named 'forecasting_evaluation.runner'`
> even though the file exists). A dedicated environment avoids this entirely.
> The same applies to any stray `PYTHONPATH` that points at another such repo —
> keep it unset inside the environment.

## Canonical install (conda for Python, pip for the rest)

The recommended pattern: use **conda only to pin the Python interpreter**, then
use **pip for every actual dependency**. This keeps the dependency set defined
in one place (`pyproject.toml`) and reproducible across machines.

```bash
git clone https://github.com/AshleyLab/myheartcounts-dataset.git
cd myheartcounts-dataset

# 1. conda provisions only the interpreter + base runtime
conda create -n openmhc python=3.10 -y
conda activate openmhc

# 2. pip installs openmhc and all dependencies
pip install -e ".[all]"        # all three tracks + Hydra CLIs + W&B logging
```

For contributor tooling (jupyter, pytest, ruff) add the `dev` extra:

```bash
pip install -e ".[all,dev]"
```

### Alternative: plain venv (no conda)

If you'd rather not use conda, a standard-library virtual environment works the
same way — it just needs a Python ≥ 3.10 to build from:

```bash
python3.10 -m venv ~/envs/openmhc      # or any Python ≥ 3.10 on your PATH
source ~/envs/openmhc/bin/activate
pip install -e ".[all]"
```

## Extras

Install only what you need — each extra is additive.

| Extra | Pulls in | Needed for |
|---|---|---|
| *(none)* | core: numpy, pandas, datasets, scikit-learn, torch, xgboost, … | Track 1 (outcome prediction); the public `evaluate_*` API surface |
| `pypots` | `pypots` (+ `pygrinder`, `tsdb`) | Tracks 2 & 3 deep-learning imputers/forecasters |
| `lsm2` | `pytorch-lightning` | LSM2 / Lightning-based models |
| `chronos` | `chronos-forecasting` | Track 3 `Chronos2Forecaster` (Chronos-2 foundation model) |
| `toto` | `toto-ts` | Track 3 `TotoForecaster` (Toto foundation model) |
| `hydra` | `hydra-core`, `omegaconf`, `hydra-submitit-launcher` | the `mhc-impute-eval` / `mhc-forecast-eval` (and `mhc-impute-train` / `mhc-forecast-train`) CLIs |
| `hf` | `huggingface_hub` | Hub-backed checkpoint/artifact downloads |
| `wandb` | `wandb` | W&B logging in the imputation pipeline |
| `all` | every runtime extra above | the full benchmark (all tracks + CLIs) |
| `dev` | jupyterlab, ipywidgets, pytest, ruff | development / running the notebooks |

## Point the API at a dataset

The eval API has **no default cache location** — you must set `MHC_DATA_DIR`
(or pass `data_dir=` per call):

```bash
export MHC_DATA_DIR=/path/to/openmhc/data-full   # the dataset root
```

Each root must contain a `dataset_version.json` marker. See
[`manual-dataset-setup.md`](manual-dataset-setup.md) for the full layout and how
to obtain / lay out the data.

## Verify

```bash
# All track engines import from THIS repo (not a shadowing copy):
python - <<'PY'
import importlib
for m in ["openmhc", "pypots", "pytorch_lightning", "hydra", "wandb",
          "imputation_evaluation", "forecasting_evaluation", "downstream_evaluation"]:
    mod = importlib.import_module(m)
    print(f"OK  {m:24s} -> {getattr(mod, '__file__', '(namespace)')}")
PY
```

A quick end-to-end check against a real dataset root (tiny sample budget):

```python
import numpy as np, openmhc

class LastValueForecaster:
    def predict(self, history, horizon):
        last = np.nan_to_num(history[:, -1:], nan=0.0)
        return np.tile(last, (1, horizon)).astype(np.float32)

res = openmhc.evaluate_forecasting(
    LastValueForecaster(), version="xs", forecasting_length=24, max_samples=5
)
print(res.summary())
```

## Sherlock (Stanford cluster) — venv, no conda

On Sherlock, conda is discouraged and the system Python builds need specific
shared libraries at runtime. Use the bundled helper instead of the canonical
recipe above:

```bash
# One-time: create the venv from Sherlock's python build
LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:/share/software/user/open/openssl/3.0.7/lib64" \
  /share/software/user/open/python/3.12.1/bin/python3 -m venv "$SCRATCH/envs/openmhc"

# Every session: activate (sets LD_LIBRARY_PATH, unsets the shell's PYTHONPATH
# and MPI compiler wrappers, then activates the venv)
source scripts/dev/activate-openmhc.sh

# Then install as usual
pip install -e ".[all]"
```

See [`scripts/dev/activate-openmhc.sh`](../scripts/dev/activate-openmhc.sh) for
details. The Track-2 cluster bootstrap (W&B artifact downloads, sanity checks)
lives in [`jobs/sherlock/imputation_eval/00_setup.sh`](../jobs/sherlock/imputation_eval/00_setup.sh).
