# Neural imputers — PyPOTS and LSM2

`openmhc.imputers` ships two families of pre-trained neural imputer wrappers on top of
the duck-typed [`Imputer`](../src/openmhc/_protocols.py) protocol used by
`openmhc.evaluate_imputation`:

- **PyPOTS family** — wrappers around four published PyPOTS imputation models
  (BRITS, TimesNet, DLinear, FEDformer). Backed by the [PyPOTS][pypots] package.
- **LSM2 family** — wrappers around the in-house masked-autoencoder ViT for 1D
  wearable data (formerly called "MAE" in the private companion repo). Two
  shapes: a standard daily/weekly variant and a weekly-sparse variant with a
  per-day encoder + cross-day decoder.

Both families share the same release-bundle format and a `from_release()`
classmethod, so loading a paper-faithful checkpoint is one line.

[pypots]: https://github.com/WenjieDu/PyPOTS

---

## Install

The neural wrappers are optional extras so the bulk of `openmhc` (mean / mode /
linear / locf / personalized / TorchImputer) doesn't need a heavy ML stack:

```bash
pip install 'openmhc[pypots]'   # for BRITS / TimesNet / DLinear / FEDformer
pip install 'openmhc[lsm2]'     # for LSM2Imputer / LSM2WeeklySparseImputer
# or both at once
pip install 'openmhc[pypots,lsm2]'
```

`pypots` pulls in `pypots>=1.2`. `lsm2` pulls in `pytorch-lightning>=2.0`.
`torch` is already a base dependency, so no extra install is needed for it.

Importing the wrapper classes (`from openmhc.imputers import LSM2Imputer`) is
always safe — the heavy deps are imported lazily inside the constructor so the
import-time surface stays minimal.

---

## Quickstart — evaluate a paper checkpoint

```python
import openmhc
from openmhc.imputers import BRITSImputer

imputer = BRITSImputer.from_release("hf://MyHeartCounts/openmhc-brits-imp")
results = openmhc.evaluate_imputation(imputer, version="xs")
print(results.summary())
```

`from_release(...)` accepts three forms:

- A local release directory (`"path/to/openmhc-brits-paper/"`)
- A direct path to a manifest file
- A `hf://org/repo[@revision]` URI for a bundle on the Hugging Face Hub

It reads the bundle's manifest, validates that its `kind` matches the
wrapper class, and constructs the imputer with the recorded architecture
hyperparameters and normalization stats. The bundle layout is identical
across PyPOTS, LSM2, and local-vs-HF storage.

Runtime knobs (`device`, `inference_batch_size`, `data_dir`, and for LSM2
`inference_dropout_removal_ratio`) can be passed as kwargs:

```python
LSM2Imputer.from_release(
    "path/to/openmhc-lsm2-daily/",
    device="cuda:0",
    inference_batch_size=128,
    inference_dropout_removal_ratio=0.0,
)
```

For batch evaluation, sweeps, or SLURM-dispatched runs against published paper
checkpoints, use the Hydra CLI documented in
[`src/imputation_evaluation/README.md`](../src/imputation_evaluation/README.md#part-15--reproducible-runs-via-mhc-impute-eval):

```bash
mhc-impute-eval method=brits method.release_dir=path/to/openmhc-brits-paper/
mhc-impute-eval method=lsm2  method.release_dir=path/to/openmhc-lsm2-daily/
mhc-impute-eval --multirun method=brits,timesnet,dlinear,fedformer \
  method.release_dir=releases/${method.type}
```

The CLI copies the release manifest into the run dir so each result is
traceable to its exact checkpoint and arch.

---

## Hugging Face Hub bundles

Paper-faithful checkpoints are mirrored on the Hugging Face Hub under the
[`MyHeartCounts`](https://huggingface.co/MyHeartCounts) organization. The
`hf://` URI scheme is the recommended path for users who don't have W&B
access; bundles are public and licensed [OpenRAIL][openrail].

```bash
pip install 'openmhc[pypots,hf]'      # PyPOTS wrappers + HF loader
pip install 'openmhc[lsm2,hf]'        # LSM2 wrappers + HF loader
```

| Release | HF repo | Wrapper |
|---|---|---|
| BRITS (daily) | `MyHeartCounts/openmhc-brits-imp` | `BRITSImputer` |
| DLinear (daily) | `MyHeartCounts/openmhc-dlinear-imp` | `DLinearImputer` |
| DLinear (7-day) | `MyHeartCounts/openmhc-dlinear-7day-imp` | `DLinearImputer` |
| FEDformer (daily) | `MyHeartCounts/openmhc-fedformer-imp` | `FEDformerImputer` |
| TimesNet (daily) | `MyHeartCounts/openmhc-timesnet-imp` | `TimesNetImputer` |
| LSM2 (daily) | `MyHeartCounts/openmhc-lsm2-daily` | `LSM2Imputer` |
| LSM2 (weekly) | `MyHeartCounts/openmhc-lsm2-weekly` | `LSM2Imputer` |
| LSM2 (weekly-sparse) | `MyHeartCounts/openmhc-lsm2-weekly-sparse` | `LSM2WeeklySparseImputer` |

The `-imp` suffix on the PyPOTS rows flags those bundles as imputation-only
fine-tunes. The LSM2 bundles are general-purpose self-supervised encoders
that happen to support imputation via the wrapper.

Pin a specific revision (once tagged) with `@`:

```python
BRITSImputer.from_release("hf://MyHeartCounts/openmhc-brits-imp@v1.0")
```

Snapshots cache via `huggingface_hub`'s default location
(`~/.cache/huggingface/hub`, controllable via `HF_HOME`). Only the
manifest, normalization stats, FEDformer `fourier_modes.json` sidecar, and
checkpoint files are downloaded — the model card and any other repo metadata
are skipped.

[openrail]: https://huggingface.co/blog/open_rlhf

---

## Release bundle layout

A release is a self-contained directory:

```
my-release/
├── model.pypots                 # or model.ckpt for LSM2
├── normalization_stats.json
├── fourier_modes.json           # required for FEDformer spec v2 bundles
└── openmhc_manifest.json
```

`openmhc_manifest.json` schema (current writer emits spec_version 2):

```json
{
  "spec_version": 2,
  "kind": "fedformer",
  "checkpoint": "model.pypots",
  "normalization_stats": "normalization_stats.json",
  "fourier_modes": "fourier_modes.json",
  "arch": {
    "n_steps": 1440,
    "n_features": 19,
    "n_layers": 2,
    "d_model": 512,
    "n_heads": 8,
    "d_ffn": 128,
    "moving_avg_window_size": 25,
    "dropout": 0.1,
    "variant": "Fourier",
    "modes": 32,
    "mode_select": "random"
  },
  "provenance": {
    "wandb_artifact": "MHC_Dataset/.../fedformer:v0",
    "wandb_run": "...",
    "val_mae": 0.0945,
    "val_epoch": 5
  }
}
```

`kind` ∈ `{brits, timesnet, dlinear, fedformer, lsm2, lsm2_weekly_sparse}`.
Paths are stored *relative to the manifest file*, so the bundle is movable —
`shutil.copytree` it anywhere and `from_release` still works. The
`normalization_stats` field may be `null` for checkpoints trained on raw
inputs (`BRITSImputer` defaults to no normalization in this project).
The `fourier_modes` field is only valid for FEDformer and is required for
current FEDformer releases; it points to the sidecar used to restore PyPOTS
FourierBlock indices. Spec v1 manifests without this sidecar are still
loadable for legacy compatibility, but FEDformer bundles without it use the
legacy "re-draw on construct" path described below and should not be used for
paper-faithful results.

`provenance` is freeform metadata; the loader ignores unknown keys.

---

## Family A — PyPOTS imputers

Four wrapper classes, one per published PyPOTS imputer. Daily-window (1440
minute) and 7-day (10080 minute) eval are both supported; pass the matching
`n_steps` so the model is rebuilt with the right sizing.

| Class | `kind` | Architecture hyperparameters |
|---|---|---|
| `BRITSImputer` | `brits` | `rnn_hidden_size` |
| `TimesNetImputer` | `timesnet` | `n_layers`, `top_k`, `d_model`, `d_ffn`, `n_kernels`, `dropout`, `apply_nonstationary_norm` |
| `DLinearImputer` | `dlinear` | `moving_avg_window_size`, `d_model`, `individual` |
| `FEDformerImputer` | `fedformer` | `n_layers`, `d_model`, `n_heads`, `d_ffn`, `moving_avg_window_size`, `dropout`, `variant`, `modes`, `mode_select` |

All four also accept the shared kwargs: `n_steps`, `n_features`, `device`,
`inference_batch_size`, `normalization_stats_path`, `data_dir`.

### Direct construction

If you don't want a release bundle (e.g. you're loading your own training
output), construct directly:

```python
from openmhc.imputers import FEDformerImputer

imputer = FEDformerImputer(
    model_path="checkpoints/fedformer_best.pypots",
    n_steps=1440,
    n_features=19,
    n_layers=2,
    d_model=512,
    n_heads=8,
    d_ffn=128,
    moving_avg_window_size=25,
    dropout=0.1,
    version="Fourier",
    modes=32,
    mode_select="random",
    normalization_stats_path="normalization_stats.json",
)
```

The arch hyperparameters **must match the trained model** — PyPOTS's
`model.load()` performs a strict state-dict load and will raise a torch
`RuntimeError` (size mismatch) on the slightest mismatch.

`model_path` accepts a direct `.pypots` file or a directory containing one
(the first sorted match wins).

### Paper checkpoints

Pulled from W&B; sizes vary by model. The release names below match the
output of `tools/build_manifest.py` when run with `--release-name`:

| Release | W&B artifact | Size | Notes |
|---|---|---|---|
| `brits` | `MHC_Dataset/mhc-pypots-brits/brits:v19` (file `BRITS_epoch5_MAE0.0945.pypots`) | ~750 KB | `rnn_hidden_size=128`, `n_steps=1440`, val MAE 0.0945 @ ep5 |
| `dlinear` | `MHC_Dataset/mhc-pypots-dlinear/dlinear:v49` (file `DLinear_epoch2_MAE0.1335.pypots`) | ~16 MB | `d_model=256`, `moving_avg_window_size=51`, `n_steps=1440`, val MAE 0.1335 @ ep2 |
| `dlinear-7day` | `MHC_Dataset/mhc-pypots-dlinear/dlinear:v48` | 776 MB | `n_steps=10080`, epoch 16 val MAE 0.1456 |
| `fedformer` | `MHC_Dataset/mhc-pypots-fedformer/fedformer:v31` | 20 MB | Run `ouqezdi7`, val MAE 0.1706 @ ep12 |
| `timesnet` | `MHC_Dataset/mhc-pypots-timesnet/timesnet:v31` | 287 MB | Run `x9386qo6`, val MAE 0.2718 @ ep11 |

### Downloading the checkpoints

The simplest path is the W&B CLI — it downloads the entire artifact directory
(checkpoint file + any sibling metadata) into `--root`:

```bash
wandb artifact get MHC_Dataset/mhc-pypots-brits/brits:v19          --root ./brits_v19
wandb artifact get MHC_Dataset/mhc-pypots-dlinear/dlinear:v49      --root ./dlinear_v49
wandb artifact get MHC_Dataset/mhc-pypots-dlinear/dlinear:v48      --root ./dlinear_v48_7day
wandb artifact get MHC_Dataset/mhc-pypots-fedformer/fedformer:v31  --root ./fedformer_v31
wandb artifact get MHC_Dataset/mhc-pypots-timesnet/timesnet:v31    --root ./timesnet_v31
```

Then point the wrappers at the downloaded directory — `BRITSImputer` /
`TimesNetImputer` / `DLinearImputer` / `FEDformerImputer` all accept a directory
as `model_path` and pick up the first `*.pypots` file inside.

To stage the checkpoint into a release bundle (with manifest + normalization
stats) for the leaderboard, feed the downloaded directory to
[`tools/build_manifest.py`](#bundling-a-release-for-the-leaderboard-toolsbuild_manifestpy)
via `--model-path-override`.

> **Don't use `:latest`.** W&B's `:latest` alias is mutable and will drift as
> new versions are pushed. Always pin to the explicit `:vNN` version shown in
> the table.

---

## Family B — LSM2 imputers

`LSM2` is the public-facing name for what the private MHC-benchmark repo calls
MAE — a masked-autoencoder vision transformer for 1D wearable data. Two
wrapper classes:

| Class | `kind` | Model class | Notes |
|---|---|---|---|
| `LSM2Imputer` | `lsm2` | `openmhc.models.lsm2.LSM2ViT1D` | Daily (`seq_length=1440`, `patch_size=10`) and weekly (`seq_length=10080`, `patch_size=60`) — same wrapper, different sizing. |
| `LSM2WeeklySparseImputer` | `lsm2_weekly_sparse` | `openmhc.models.lsm2.WeeklySparseDecoderLSM2` | Per-day frozen encoder + sparse cross-day decoder for 7-day windows. |

### Architecture hyperparameters

Shared across both classes (specify whichever match the trained model):

- `seq_length`, `patch_size`, `in_channels`
- `embed_dim`, `depth`, `num_heads`
- `decoder_embed_dim`, `decoder_depth`, `decoder_num_heads`
- `mlp_ratio`, `mask_ratio`

`LSM2WeeklySparseImputer` additionally takes:

- `num_days` (default 7), `window_minutes` (default 120),
  `use_rope_day_embed` (default True), `freeze_encoder` (default True)

All wrappers also accept the shared kwargs: `device`, `inference_batch_size`,
`inference_dropout_removal_ratio`, `normalization_stats_path`, `data_dir`.

### Direct construction

```python
from openmhc.imputers import LSM2Imputer

imputer = LSM2Imputer(
    model_path="checkpoints/lsm2-daily.ckpt",   # Lightning .ckpt
    seq_length=1440,
    patch_size=10,
    embed_dim=384,
    depth=12,
    num_heads=6,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=4,
    mlp_ratio=4.0,
    mask_ratio=0.5,
    normalization_stats_path="normalization_stats.json",
)
```

A note on the arch kwargs: LSM2 checkpoints are loaded via PyTorch Lightning's
`load_from_checkpoint`, which restores the model from the **saved hparams**
inside the checkpoint, not from the kwargs you pass here. So mismatched values
won't crash the load — they only affect the manifest's recorded `arch` block.
The wrapper records what you supplied for documentation, but the real
architecture comes from the checkpoint.

### Inference flow

1. Z-score channels 0–6 with the sibling `normalization_stats.json`.
2. Fill remaining NaNs (naturally missing + target positions) with 0 — the
   channel mean after normalization.
3. Build a patch-level inherited mask: a patch is "missing" (1) if **any**
   minute in it is missing, except the HR channel (index 5), where the rule
   is **all** minutes missing.
4. Run a custom inference forward pass that bypasses training-time artificial
   masking (`total_mask = inherited_mask`). Prioritized-keep ordering uses
   `inference_dropout_removal_ratio` (override the checkpoint's value with
   `inference_dropout_removal_ratio=0.0` for deterministic inference).
5. Unpatchify, sigmoid the binary channels if `model.use_hybrid_loss=True`,
   denormalize.
6. Write predictions back only at `target_mask == 1` positions.

### Paper checkpoints

| Release | W&B artifact | Wrapper |
|---|---|---|
| `lsm2-daily` | `MHC_Dataset/mhc-mae-ssl-daily/mae-daily:v0` | `LSM2Imputer` |
| `lsm2-weekly` | `MHC_Dataset/mhc-mae-ssl/model-o5quh2cd:v2` | `LSM2Imputer` with `seq_length=10080`, `patch_size=60` |
| `lsm2-weekly-sparse` | `MHC_Dataset/mhc-mae-ssl/mae-weekly-sparse-d4:v0` | `LSM2WeeklySparseImputer` |

The daily and weekly bundles share `LSM2Imputer` — the wrapper handles both
by reading the trained sizing from the checkpoint.

### Downloading the checkpoints

Same W&B CLI pattern as the PyPOTS family:

```bash
wandb artifact get MHC_Dataset/mhc-mae-ssl-daily/mae-daily:v0           --root ./lsm2_daily_v0
wandb artifact get MHC_Dataset/mhc-mae-ssl/model-o5quh2cd:v2            --root ./lsm2_weekly_v2
wandb artifact get MHC_Dataset/mhc-mae-ssl/mae-weekly-sparse-d4:v0      --root ./lsm2_weekly_sparse_v0
```

The LSM2 wrappers look for `*.ckpt` (then `*.pt`, then `*.pth`) inside the
downloaded directory. Direct wrapper construction only reads a
`normalization_stats.json` passed via `normalization_stats_path`; it does not
extract stats from the checkpoint by itself. The release converter can extract
stats embedded in `ckpt["LightningDataModule"]["normalization_stats"]` and
write the sibling JSON into the bundle, which is what `from_release()` then
uses.

---

## Bundling a release for the leaderboard (`tools/build_manifest.py`)

Given a private-repo eval config and a `.pypots` (or `.ckpt`) on disk, the
converter stages a release directory:

```bash
# PyPOTS — config has all arch fields inline; stats are a sibling JSON
python tools/build_manifest.py \
  --source-repo /path/to/private-repo \
  --output-dir releases/ \
  --config /path/to/private-repo/configs/imputation_eval/methods/pypots_fedformer.yaml \
  --model-path-override /path/to/fedformer_v31/FEDformer.pypots \
  --stats-path-override /path/to/private-repo/data/processed/normalization_stats.json \
  --release-name fedformer --overwrite
```

For LSM2, the converter knows how to extract architecture hyperparameters
from `ckpt["hyper_parameters"]` and normalization stats from
`ckpt["LightningDataModule"]["normalization_stats"]`, so no `--stats-path-override`
is needed (it falls back to the YAML pointer if those keys are absent):

```bash
python tools/build_manifest.py \
  --source-repo /path/to/private-repo \
  --output-dir releases/ \
  --config /path/to/private-repo/configs/imputation_eval/methods/mae.yaml \
  --model-path-override /path/to/lsm2-daily.ckpt \
  --release-name lsm2-daily --overwrite
```

The converter accepts the private repo's old `method.type: mae` and
`method.type: mae_weekly_sparse` values and maps them to the public
`lsm2` / `lsm2_weekly_sparse` `kind` automatically.

Pulling artifacts first:

```bash
wandb artifact get MHC_Dataset/mhc-mae-ssl-daily/mae-daily:v0 \
  --root ~/wandb-cache/lsm2-daily
```

Flags worth knowing:

- `--overwrite` — replace an existing release directory.
- `--no-copy-checkpoint` — emit the manifest only, with a relative path to the
  original checkpoint. Useful for testing without duplicating 800 MB weight files.
- `--model-path-override`, `--stats-path-override` — only valid with a single
  `--config`; override the YAML-recorded paths (mandatory after `wandb artifact get`
  since the YAML's `wandb:...` URI doesn't resolve on disk).

---

## Authoring a release manifest by hand

If you don't want to use the converter (e.g. you trained your own model
outside the private-repo workflow), write a manifest directly:

```python
from openmhc.imputers import write_manifest

write_manifest(
    "releases/my-brits/",
    kind="brits",
    arch={"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
    checkpoint="model.pypots",
    normalization_stats="normalization_stats.json",
    provenance={"trained_by": "my-team", "git_commit": "abc123"},
)
```

Then drop `model.pypots` and `normalization_stats.json` next to the manifest.
`BRITSImputer.from_release("releases/my-brits/")` loads it.

---

## Troubleshooting

**`Manifest is for kind 'brits', but TimesNetImputer expects kind 'timesnet'`** —
You called `from_release` on the wrong wrapper class. Use `BRITSImputer.from_release`
for that bundle, or check the `kind` field of `openmhc_manifest.json`.

**`RuntimeError: size mismatch for ...` on PyPOTS load** — One of the arch
hyperparameters doesn't match the trained model. Inspect the manifest's `arch`
block and pass those exact values to the constructor. The most common offender
is `rnn_hidden_size` (BRITS) or `d_model` (everything else).

**`Stats file referenced by config does not exist locally`** (converter warning) —
The eval config points at a path that's missing in your source repo. The
converter still emits a manifest with `normalization_stats: null`, but the
loaded wrapper will run without normalization — almost always wrong for LSM2 and
PyPOTS models trained on z-scored inputs. Pass `--stats-path-override` or copy
the stats file before re-running.

**`ModuleNotFoundError: No module named 'pypots'` / `'pytorch_lightning'`** —
Install the optional extra: `pip install 'openmhc[pypots]'` or
`pip install 'openmhc[lsm2]'`.

**`No .ckpt / .pt / .pth checkpoint found under directory ...`** — LSM2 wrappers
look for `*.ckpt` first, then `*.pt`, then `*.pth`. If your checkpoint has a
different extension, pass the file path directly instead of the parent dir, or
rename it.

**The wrapper imports succeed but the model class hasn't been ported.** Only
four PyPOTS models (BRITS, TimesNet, DLinear, FEDformer) and the two LSM2
variants (`LSM2ViT1D`, `WeeklySparseDecoderLSM2`) ship public wrappers. SAITS,
TimeMixer, TRMF, and others remain private — `from openmhc.imputers import SAITSImputer`
will fail with `ImportError`.

---

## Training your own imputer

The companion package `imputation_training` ships a full Hydra CLI
(`mhc-impute-train`) for training any of the four supported PyPOTS
models on the OpenMHC dataset. The output is a release-bundle directory
directly consumable by the eval CLI's `method.release_dir=...` flag.

Quick start:

```bash
mhc-impute-train \
    model=fedformer \
    seed=42 \
    data.version=full \
    data.daily_hf_dir=/path/to/daily_hf \
    +data.split_file=/path/to/sharable_users.json \
    training.epochs=50 \
    output.release_dir=/path/to/my-fedformer-release

# The trained bundle is consumable by mhc-impute-eval immediately:
mhc-impute-eval method=fedformer method.release_dir=/path/to/my-fedformer-release ...
```

The same CLI handles BRITS, DLinear, and TimesNet — just swap
`model=fedformer` for the model name. See
[`src/imputation_training/README.md`](../src/imputation_training/README.md)
for the public Python API, [`configs/training/`](../configs/training/)
for the YAML schema, and
[`jobs/sherlock/imputation_train/README.md`](../jobs/sherlock/imputation_train/README.md)
for the SLURM walkthrough.

### Why FEDformer training needs special handling

PyPOTS's `FourierBlock` contains an upstream bug: each instance calls
`np.random.shuffle` at construction time to pick which frequency bins
its `weights1` parameter operates on, but the resulting index list is
stored as a plain Python attribute and **is not** saved in
`state_dict`. Loading a `.pypots` checkpoint in a fresh process
re-draws the indices against an unknown RNG state, so the trained
weights index into the wrong frequency bins (~2–6% NRMSE degradation
in our parity audit).

The `imputation_training` pipeline sidesteps this by:

1. Seeding all RNGs deterministically before model construction.
2. Capturing each trained `FourierBlock.index` to a
   `fourier_modes.json` sidecar written next to the `.pypots` file in
   the release bundle. Manifest spec v2 adds a required FEDformer
   `fourier_modes` field that points at the sidecar.
3. Restoring those indices post-`model.load()` at inference time (see
   `FEDformerImputer._post_load`).

The result: trained FEDformer models load byte-identical across
processes, machines, and Python invocations. The fix is
forward-compatible — older bundles without the sidecar still load via
the legacy "re-draw on construct" path (and exhibit the bug, as
before). Bundles published before this fix were spec v1; new bundles
authored by the training pipeline are spec v2.
