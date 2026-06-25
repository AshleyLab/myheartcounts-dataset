# Installation

How to install `openmhc` and its evaluation dependencies in a clean, isolated environment. For getting the dataset itself, see [manual-dataset-setup.md](manual-dataset-setup.md).

## Prerequisites

- **Python ≥ 3.10**
- An **isolated environment** dedicated to `openmhc` (see the warning below).

> [!WARNING]
> **Install `openmhc` into its own environment.** The evaluation engines ship top-level packages — `forecasting_evaluation`, `imputation_evaluation`, `downstream_evaluation`, `forecasting_training`, `imputation_training`, `eval_hydra`, `labels`, `data`, `context`, `devices`, `utils` — whose names are **not unique** to this repo. The private `MHC-benchmark` repo defines the same names. If both are installed (editable) into one environment, whichever sits earlier on `sys.path` silently shadows the other, and imports resolve to the wrong copy (e.g. `ModuleNotFoundError: No module named 'forecasting_evaluation.runner'` even though the file exists). A dedicated environment avoids this entirely. The same applies to any stray `PYTHONPATH` that points at another such repo — keep it unset inside the environment.

## Canonical install (reproducible / pinned)

Use **conda only to pin the Python interpreter**, then **pip with the version freeze** (`constraints.txt`) for every dependency. The freeze *is* the environment that produced the published results — installing against it is what makes a fresh machine reproduce our numbers (CPU methods bit-exact) rather than resolving to whatever PyPI serves that day. `constraints.txt` documents the platform caveats (bit-exactness holds on linux-x86_64 / CPython 3.10 / pip).

```bash
git clone https://github.com/AshleyLab/myheartcounts-dataset.git
cd myheartcounts-dataset

# 1. conda provisions only the interpreter
conda create -n openmhc python=3.10 -y
conda activate openmhc

# torch is pinned to a CUDA build, so every pinned install needs the cu126 index:
IDX="--extra-index-url https://download.pytorch.org/whl/cu126"

# 2. openmhc + 6 of 8 methods (linear, multirocket, xgboost, gru_d, lsm2, chronos2), frozen.
#    [all] intentionally excludes toto + wbm — they need the two extra steps below.
pip install -e ".[all]" -c constraints.txt $IDX

# 3. toto: toto-ts over-pins its transitive deps (torch/numpy/datasets), so install the
#    wheel WITHOUT its deps, then supply only the runtime deps its forward needs.
pip install --no-deps toto-ts==0.2.0
pip install -e ".[toto_dep]" -c constraints.txt $IDX

# 4. wbm: CUDA-kernel build from source — see "Building wbm / mamba-ssm" below.
```

That covers all 8 methods. For contributor tooling (jupyter, pytest, ruff) add the `dev` extra to step 2: `pip install -e ".[all,dev]" -c constraints.txt $IDX`.

> [!NOTE]
> **On an HPC cluster, export `HF_HUB_DISABLE_XET=1`** before any checkpoint pull — the Hugging Face Xet download backend can crash on cluster / parallel filesystems.

For a **quick, unpinned** install (latest PyPI; *not* guaranteed to reproduce the published numbers), drop the `-c constraints.txt $IDX` suffix: `pip install -e ".[all]"`.

### Alternative: plain venv (no conda)

If you'd rather not use conda, a standard-library virtual environment works the same way — it just needs a Python ≥ 3.10 to build from:

```bash
python3.10 -m venv ~/envs/openmhc      # or any Python ≥ 3.10 on your PATH
source ~/envs/openmhc/bin/activate
# then the same pinned steps 2–4 as above (with -c constraints.txt $IDX)
```

## Extras

Install only what you need — each extra is additive.

| Extra | Pulls in | Needed for |
|---|---|---|
| *(none)* | core: numpy, pandas, datasets, scikit-learn, torch, xgboost, … | Track 1 (outcome prediction); the public evaluate_* API surface |
| `pypots` | `pypots` (+ `pygrinder`, `tsdb`) | Tracks 2 & 3 deep-learning imputers/forecasters |
| `lsm2` | `pytorch-lightning` | LSM2 / Lightning-based models |
| `chronos` | `chronos-forecasting` | Track 3 `Chronos2Forecaster` (Chronos-2 foundation model) |
| `toto` | `toto-ts` | Track 3 `TotoForecaster` (Toto foundation model) |
| `wbm` | `mamba-ssm`, `causal-conv1d` (CUDA-compiled) | `wbm` Mamba-2 week encoder — **needs a from-source CUDA build**, see [Building wbm / mamba-ssm](#building-wbm--mamba-ssm-from-source) |
| `hydra` | `hydra-core`, `omegaconf`, `hydra-submitit-launcher` | the `mhc-downstream-eval` / `mhc-impute-eval` / `mhc-forecast-eval` (+ `mhc-impute-train` / `mhc-forecast-train`) CLIs |
| `hf` | `huggingface_hub` | Hub-backed checkpoint/artifact downloads — **required by `toto` and `chronos2`** (they fetch their checkpoints from the Hub); included in `all` |
| `wandb` | `wandb` | W&B logging in the imputation pipeline |
| `all` | every runtime extra above | the full benchmark (all tracks + CLIs) |
| `dev` | jupyterlab, ipywidgets, pytest, ruff | development / running the notebooks |

## Building `wbm` / mamba-ssm from source

The `wbm` method's Mamba-2 week encoder depends on `mamba-ssm` + `causal-conv1d`, which are **CUDA-kernel** packages — not pure Python. They can't be a clean pip extra, for two reasons:

- their **prebuilt wheels** are built against a recent glibc (≥ 2.32) and a fixed torch/CUDA/Python/ABI, so on many HPC systems (e.g. RHEL/Rocky 8 = glibc 2.28) they fail to load with `ImportError: ... GLIBC_2.32 not found`;
- `mamba-ssm`'s package metadata **force-upgrades `torch`** and pulls heavy build backends, which corrupts the pinned environment.

So `wbm` is installed by **compiling the kernels from source**. This needs a **CUDA toolkit (`nvcc`) matching your torch CUDA build** and **gcc ≥ 11**.

```bash
# 1. Make nvcc (matching torch's CUDA, e.g. 12.x) and gcc >= 11 available. On an HPC
#    cluster these are usually environment modules. Example (Imperial RDS):
module load tools/prod GCC/12.3.0 CUDA/12.6.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

# 2. After `pip install -e ".[all]"` (so torch is already present), build from the GitHub
#    SOURCES. Build from git, not PyPI: the PyPI sdists omit the C++/CUDA code under
#    csrc/, so a `--no-binary` install from PyPI fails with "csrc/...cpp ... missing".
#    --no-deps : don't let mamba-ssm's metadata upgrade torch / pull extras
#    TORCH_CUDA_ARCH_LIST : YOUR GPU's compute capability
#       (L40S 8.9 · A100 8.0 · H100 9.0 · V100 7.0 · RTX 30xx 8.6)
MAX_JOBS=8 \
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST=8.9 \
  pip install --no-build-isolation --no-deps \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.4"

# 3. Verify (run on a GPU node):
python -c "from mamba_ssm import Mamba2; print('mamba-ssm OK')"
```

Run the build on a node that has both `nvcc` **and** a GPU, so the kernels compile for the right architecture and the import check can load them.

> [!NOTE]
> If your platform matches a published wheel (recent glibc + your exact torch/CUDA/Python), you can instead install the matching `.whl` from the [state-spaces/mamba](https://github.com/state-spaces/mamba/releases) and [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) releases — but the source build above is the portable path.

## Point the API at a dataset

The eval API has **no default cache location** — you must set `MHC_DATA_DIR` (or pass `data_dir=` per call):

```bash
export MHC_DATA_DIR=/path/to/openmhc/data-full   # the dataset root
```

Each root must contain a `dataset_version.json` marker. See [manual-dataset-setup.md](manual-dataset-setup.md) for the full layout and how to obtain / lay out the data.

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

On Sherlock, conda is discouraged and the system Python builds need specific shared libraries at runtime. Use the bundled helper instead of the canonical recipe above:

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

See [scripts/dev/activate-openmhc.sh](../scripts/dev/activate-openmhc.sh) for details. The Track-2 cluster bootstrap (W&B artifact downloads, sanity checks) lives in [jobs/sherlock/imputation_eval/00_setup.sh](../jobs/sherlock/imputation_eval/00_setup.sh).
