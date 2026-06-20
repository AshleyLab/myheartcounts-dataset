# Sherlock imputer-training runner

End-to-end SLURM workflow for training PyPOTS imputers (BRITS, DLinear,
TimesNet, FEDformer) on the OpenMHC dataset, producing release bundles
the eval CLI consumes directly.

The canonical worked example is **retraining FEDformer with the
FourierBlock-index sidecar fix** — see `run_fedformer_train.sbatch`.
The same pipeline works for the other three models by passing
`model=brits|dlinear|timesnet` to `mhc-impute-train`.

## Layout

```
jobs/sherlock/imputation_train/
├── _common.sh                       # sourced env (venv + paths)
├── run_fedformer_train.sbatch       # FEDformer retrain (1-2 GPU-days)
└── README.md                        # this file
```

## Why this exists

PyPOTS' `FourierBlock` (used by FEDformer) calls `np.random.shuffle` at
construction time and stores the resulting frequency-mode indices on a
plain Python attribute that **is not** saved in the model `state_dict`.
Loading a `.pypots` checkpoint in a fresh process re-draws the indices
against an unknown `np.random` state — the trained `weights1` then
operates on the wrong frequency bins (we measured ~2–6% NRMSE
degradation on the OpenMHC parity audit).

The OpenMHC training pipeline (`src/imputation_training/`) sidesteps
this by:

1. Seeding `random`, `numpy`, and `torch` **before** model construction
   (see `imputation_training.seeding.seed_everything`).
2. Extracting each trained `FourierBlock.index` value and writing them
   to a `fourier_modes.json` sidecar in the release bundle (see
   `imputation_training.release.write_release`).
3. Restoring those indices post-`model.load()` at inference time (see
   `openmhc.imputers.pypots.FEDformerImputer._post_load`).

The result: trained FEDformer models load byte-identical across
processes, machines, and Python invocations.

## Prerequisites

1. **OpenMHC venv** somewhere on `$SCRATCH` (same one the eval pipeline
   uses; `scripts/dev/activate-openmhc.sh` activates it).
2. **Dataset cache** under `${MHC_CACHE}` (defaults to
   `${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-full/`) with
   `processed/daily_hf/`, `processed/normalization_stats.json`, and
   `splits/sharable_users_seed42_2026.json`. Same cache the eval pipeline
   reads.
3. **GPU access** to the `gpu` partition. The sbatch requests
   Tesla-family cards explicitly to avoid the consumer-RTX-3090
   "GPU is lost" failures we hit during eval.

## How to use

### Train FEDformer (the worked example)

```bash
sbatch jobs/sherlock/imputation_train/run_fedformer_train.sbatch
```

That submits one ~48h GPU job. Watch progress with `squeue -u $USER`
and `tail -f ${LOGS}/train_fedformer_*.out` (defaults to
`/scratch/users/$USER/logs/openmhc/train_fedformer_*.out`).

When training finishes, a release bundle appears at
`${TRAIN_BASE}/releases/fedformer_<timestamp>/` (defaults to
`${SCRATCH_RUN_ROOT}/openmhc-imputation-train/releases/...`) containing:

```
fedformer_<timestamp>/
├── model.pypots              # trained weights
├── normalization_stats.json  # copied from the dataset cache
├── fourier_modes.json        # ← the per-FourierBlock indices that fix the bug
└── openmhc_manifest.json     # spec_version=2
```

### Swap into the eval pipeline + verify parity

```bash
# Point the FEDformer eval sbatch at the new release dir
sed -i \
  "s|method.release_dir=\${RELEASES}/fedformer|method.release_dir=${TRAIN_BASE}/releases/fedformer_<timestamp>|" \
  jobs/sherlock/imputation_eval/run_fedformer.sbatch

# Re-run FEDformer eval
sbatch jobs/sherlock/imputation_eval/run_fedformer.sbatch

# Once that completes
python jobs/sherlock/imputation_eval/verify_parity.py --methods fedformer
```

Expected outcome: all FEDformer rows fall within the strict 1%
tolerance against the MHC-benchmark reference (vs the 2–6% spread we
observed with the original buggy checkpoint).

### Train a different model

The `mhc-impute-train` CLI accepts any of `model=brits|dlinear|timesnet|fedformer`.
For example, to train DLinear:

```bash
source jobs/sherlock/imputation_train/_common.sh
mhc-impute-train \
  model=dlinear \
  seed=42 \
  data.version=full \
  data.daily_hf_dir="${DAILY_HF_DIR}" \
  +data.split_file="${SPLIT_FILE}" \
  training.epochs=50 \
  output.release_dir=${RELEASES_ROOT}/dlinear_$(date +%Y%m%d_%H%M%S)
```

(Wrap that in a `run_dlinear_train.sbatch` if you want SLURM dispatch —
copy `run_fedformer_train.sbatch` and change `model=...`.)

## Smoke test (interactive, ~5 min on `sh_dev`)

Before committing 48h of GPU time, validate the pipeline end-to-end on
the tiny `xs` dataset:

```bash
sh_dev -t 1:00:00 -p gpu --gres=gpu:1
source jobs/sherlock/imputation_train/_common.sh
export MHC_DATA_DIR=${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-xs

mhc-impute-train \
  model=fedformer \
  seed=42 \
  data=default \
  data.version=xs \
  data.daily_hf_dir=${MHC_DATA_DIR}/processed/daily_hf \
  +data.split_file=${MHC_DATA_DIR}/splits/sharable_users_seed42_2026_xs.json \
  data.batch_size=32 \
  h5_export.output_dir=${H5_CACHE}/xs \
  training.epochs=1 \
  training.batch_size=4 \
  output.saving_path=${RUNS_ROOT}/_smoke/fedformer_xs \
  output.release_dir=${RELEASES_ROOT}/_smoke/fedformer_xs

# Verify the FourierBlock-restore round-trip works
python -c "
from openmhc.imputers import FEDformerImputer
imp1 = FEDformerImputer.from_release('${RELEASES_ROOT}/_smoke/fedformer_xs', version='xs', device='cpu')
imp2 = FEDformerImputer.from_release('${RELEASES_ROOT}/_smoke/fedformer_xs', version='xs', device='cpu')
got = {name: m.index for name, m in imp1._model.model.named_modules() if type(m).__name__ == 'FourierBlock'}
got2 = {name: m.index for name, m in imp2._model.model.named_modules() if type(m).__name__ == 'FourierBlock'}
assert got == got2, 'Sidecar restore inconsistent across loads!'
print(f'OK: {len(got)} FourierBlocks load identical indices across processes')
"
```

If the smoke test passes, submit the full job with confidence.

## Resource notes

| field | value | rationale |
|---|---|---|
| partition | `gpu` | only place with GPU access on Sherlock |
| time | `48:00:00` | Sherlock `gpu` partition cap; paper run on Stanford simurgh logged its checkpoint at epoch 12 (well under 48h) |
| cpus-per-task | `8` | DataLoader workers + H5 export parallelism |
| mem | `64G` | matches the paper run's sbatch (Stanford simurgh) |
| gpus | `1` | PyPOTS doesn't support multi-GPU |
| constraint | `GPU_BRD:TESLA` | Paper config peaks at ~10 GB VRAM (probed via `scripts/dev/_fedformer_vram_probe.py`), so any Tesla GPU >=16 GB fits. Avoiding consumer RTX cards which cause silent requeues. |

## Paper-checkpoint provenance

The values used by `run_fedformer_train.sbatch` (model arch in
`configs/training/model/fedformer.yaml`, training hyperparams in
`configs/training/training/fedformer_paper.yaml`) come from the W&B run
that produced the published artifact `fedformer:v31`:

- Run: `MHC_Dataset/mhc-pypots-fedformer/runs/ouqezdi7` (created
  2026-03-10, standalone — not part of any sweep)
- Best val MAE: 0.1706 logged at epoch 12

Notable: the W&B run's hyperparameters were CLI overrides at training
time and do **not** match either snapshot of MHC-benchmark's
`configs/pypots/fedformer.yaml` (whose `d_ffn`/`modes`/`batch_size`
were different both before and after the paper run). Anyone reading the
old repo's `configs/pypots/fedformer.yaml` to reproduce the bundle will
be misled — the W&B run is the only authoritative source.

## Open items (not blockers)

- W&B artifact upload is opt-in via `output.wandb_enabled=true`; off by
  default. If you want the trained `.pypots` file pushed to a W&B
  artifact registry, set `output.wandb_project` and
  `output.wandb_entity` and pass `output.wandb_enabled=true`.
- This pipeline trains a single seed per run. If you want
  multi-seed ablations, use Hydra's `--multirun seed=42,43,44`.
