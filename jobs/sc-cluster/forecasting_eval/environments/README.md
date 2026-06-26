# Pinned environments — Track 3 (forecasting) experiments

Exact, fully-pinned dependency snapshots of the conda environments used to run the
forecasting evaluation on Simurgh (SC). Captured 2026-06-26 from repo commit
`76f2b36`. All three environments are **Python 3.10.20**.

The forecasting harness uses **three** conda environments (selected per model via
`MHC_CONDA_ENV` in the `run_*.sbatch` jobs; see `../_common.sh`):

| Lock file | conda env | Models | Key backend |
|---|---|---|---|
| `requirements.openmhc.txt` | `openmhc` | DLinear, MixLinear, SegRNN, autoARIMA, autoETS, naive | `pypots==1.5`, `torch==2.12.0`, `numpy==2.2.6` |
| `requirements.openmhc-toto.txt` | `openmhc-toto` | Toto | `toto-ts==0.2.0`, `torch==2.7.0`, `numpy==1.26.4` |
| `requirements.openmhc-chronos2.txt` | `openmhc-chronos2` | Chronos-2 | `chronos-forecasting==2.2.2`, `torch==2.12.0`, `numpy==2.2.6` |

Toto and Chronos-2 each need a dedicated env because their backends pin
incompatible `torch`/`numpy`/`transformers` versions relative to the `pypots`
stack in the base `openmhc` env.

## Reproduce

```bash
conda create -n openmhc python=3.10.20
conda activate openmhc
pip install -r requirements.openmhc.txt
# repeat for openmhc-toto / openmhc-chronos2 as needed
```

The `openmhc` package itself appears in each lock as an editable install pinned to
the run's repo commit (`-e git+ssh://...@76f2b36...#egg=openmhc`). Replace that line
with `pip install -e .` from your own checkout if you are not installing from the
pinned remote commit.
