# Neural forecasters — foundation fine-tunes and from-scratch baselines

`openmhc.forecasters` ships reference forecaster wrappers on top of the
duck-typed [`Forecaster`](../src/openmhc/_protocols.py) protocol used by
`openmhc.evaluate_forecasting` (Track 3). All are trained or fine-tuned on the
MHC training split and grouped into two families:

- **Foundation fine-tunes** — wrappers around two published time-series
  foundation models fine-tuned on MHC: Amazon **Chronos-2** and Datadog
  **Toto**. Each normalizes inputs internally.
- **From-scratch neural baselines** — wrappers around three PyPOTS forecasters
  trained from scratch on MHC: **DLinear**, **SegRNN**, **MixLinear**. Backed
  by the [PyPOTS][pypots] package; predictions are inverse-transformed with the
  bundled train-split `StandardScaler`.

Both families share the same release-bundle format and a `from_release()`
classmethod, so loading a paper-faithful checkpoint is one line. The bundle is
identical to the one the evaluation harness loads via `model.release_dir=…`, so
a single directory is consumable by both the public API and `mhc-forecast-eval`.

[pypots]: https://github.com/WenjieDu/PyPOTS

---

## Install

The neural-wrapper dependencies are optional extras so the bulk of `openmhc`
(plus the statistical/naive baselines) doesn't need a heavy ML stack:

```bash
pip install 'openmhc[pypots]'    # for DLinear / SegRNN / MixLinear
pip install 'openmhc[chronos]'   # for Chronos2Forecaster (the `chronos` package)
pip install 'openmhc[toto]'      # for TotoForecaster (the `toto-ts` package)
```

`pypots` pulls in `pypots>=1.2` (the floor required by the MixLinear forecaster;
it also ships the imputation models). `torch` is already a base dependency.

Add the `hf` extra to load `hf://` bundles from the Hugging Face Hub:

```bash
pip install 'openmhc[pypots,hf]'   # neural wrappers + HF loader
pip install 'openmhc[chronos,hf]'  # Chronos-2 wrapper + HF loader
```

Importing the wrapper classes (`from openmhc.forecasters import DLinearForecaster`)
is always safe — the heavy deps are imported lazily inside the constructor, so
the import-time surface stays minimal.

---

## Quickstart — evaluate a paper checkpoint

```python
import openmhc
from openmhc.forecasters import Chronos2Forecaster

fc = Chronos2Forecaster.from_release("hf://MyHeartCounts/openmhc-chronos2-fc")
results = openmhc.evaluate_forecasting(fc, version="full")
print(results.summary())
```

`from_release(...)` accepts three forms:

- A local release directory (`"releases-fc/openmhc-dlinear-fc/"`)
- A direct path to a manifest file
- A `hf://org/repo[@revision]` URI for a bundle on the Hugging Face Hub

It reads the bundle's manifest, validates that its `kind` matches the wrapper
class, and constructs the forecaster with the recorded architecture (`arch`) and
the resolved checkpoint/normalization paths. The bundle layout is identical
across both families and across local-vs-HF storage.

Runtime knobs are forwarded as kwargs (they must not duplicate any key the
manifest already records in `arch`):

```python
TotoForecaster.from_release(
    "hf://MyHeartCounts/openmhc-toto-fc",
    device="cuda:0",
    context_length=2048,
    num_samples=256,
)
```

For batch evaluation, sweeps, or SLURM-dispatched runs against published paper
checkpoints, use the Hydra CLI documented in
[`src/forecasting_evaluation/README.md`](../src/forecasting_evaluation/README.md#part-2--reproducible-runs--custom-models-via-mhc-forecast-eval-hydra):

```bash
mhc-forecast-eval model=chronos2 model.release_dir=hf://MyHeartCounts/openmhc-chronos2-fc
mhc-forecast-eval model=dlinear  model.release_dir=releases-fc/openmhc-dlinear-fc/
mhc-forecast-eval --multirun model=dlinear,segrnn,mixlinear \
  model.release_dir=releases-fc/openmhc-${model.type}-fc
```

The CLI copies the release manifest into the run dir so each result is traceable
to its exact checkpoint and arch.

---

## Hugging Face Hub bundles

Paper-faithful checkpoints are mirrored on the Hugging Face Hub under the
[`MyHeartCounts`](https://huggingface.co/MyHeartCounts) organization. The
`hf://` URI scheme is the recommended path; bundles are public and licensed
CC-BY-4.0.

| Release | HF repo | Wrapper | Family |
|---|---|---|---|
| Chronos-2 (fine-tuned) | `MyHeartCounts/openmhc-chronos2-fc` | `Chronos2Forecaster` | foundation |
| Toto (fine-tuned) | `MyHeartCounts/openmhc-toto-fc` | `TotoForecaster` | foundation |
| DLinear (from-scratch) | `MyHeartCounts/openmhc-dlinear-fc` | `DLinearForecaster` | neural |
| SegRNN (from-scratch) | `MyHeartCounts/openmhc-segrnn-fc` | `SegRNNForecaster` | neural |
| MixLinear (from-scratch) | `MyHeartCounts/openmhc-mixlinear-fc` | `MixLinearForecaster` | neural |

The `-fc` suffix flags these as forecasting (Track 3) bundles. All five target
the same task: 24-hour-ahead, 19 sensor channels, hourly resolution.

Pin a specific revision (once tagged) with `@`:

```python
Chronos2Forecaster.from_release("hf://MyHeartCounts/openmhc-chronos2-fc@v1.0")
```

Snapshots cache via `huggingface_hub`'s default location
(`~/.cache/huggingface/hub`, controllable via `HF_HOME`). Only the manifest,
checkpoint payload, and (for neural models) the `training_config.json` /
`standard_scaler_stats.json` sidecars are downloaded — the model card and other
repo metadata are skipped. The Chronos-2 bundle keeps a full HuggingFace model
directory under `checkpoint/`, so the loader's allowlist reaches into
sub-directories.

---

## Release bundle layout

A release is a self-contained directory. The payload differs by family:

```
openmhc-dlinear-fc/                 # neural (PyPOTS) — the bundle dir IS the checkpoint
├── OnlineDLinear.pypots
├── training_config.json            # architecture source of truth
├── standard_scaler_stats.json      # train-split scaler for inverse-transform
└── openmhc_manifest.json

openmhc-chronos2-fc/                 # foundation (Chronos-2) — merged full HF model
├── checkpoint/
│   ├── config.json
│   └── model.safetensors
└── openmhc_manifest.json

openmhc-toto-fc/                     # foundation (Toto) — Lightning checkpoint
├── model.ckpt
└── openmhc_manifest.json
```

`openmhc_manifest.json` schema (`spec_version` 1):

```json
{
  "spec_version": 1,
  "kind": "dlinear",
  "checkpoint": ".",
  "normalization_stats": "standard_scaler_stats.json",
  "arch": {
    "n_steps": 168,
    "n_pred_steps": 24,
    "n_features": 19
  },
  "provenance": {
    "model": "dlinear",
    "trained_on": "MHC training split (from-scratch)",
    "wandb_project": "mhc-forecasting",
    "paper_table": "tab:forecasting_grouped_model_summary"
  }
}
```

`kind` ∈ `{dlinear, segrnn, mixlinear, chronos2, toto}`. The `checkpoint` field
points at the payload **relative to the manifest file** — a file (`model.ckpt`),
a sub-directory (`checkpoint`), or `.` (the bundle dir itself, for neural models
where the internal loader reads the co-located `.pypots` + `training_config.json`).
Because paths are relative, the bundle is movable — `shutil.copytree` it anywhere
and `from_release` still works.

`normalization_stats` is `null` for the foundation models (Chronos-2 and Toto
normalize internally) and points at `standard_scaler_stats.json` for the neural
models. `arch` is splatted into the wrapper constructor; for neural models it is
deliberately minimal (`n_steps` / `n_pred_steps` / `n_features`) because the real
architecture is read from the bundled `training_config.json`. `provenance` is
freeform metadata; the loader ignores unknown keys.

This manifest is byte-compatible with
[`forecasting_evaluation.hydra.release`](../src/forecasting_evaluation/hydra/release.py),
so the same directory loads through both this public API and the evaluation
harness (`model.release_dir=…`).

---

## Family A — From-scratch neural forecasters

Three wrapper classes, one per PyPOTS forecaster trained from scratch on MHC.
The released bundle is a *directory* co-locating the `.pypots` checkpoint, the
`training_config.json` (architecture source of truth), and the
`standard_scaler_stats.json` used to inverse-transform predictions back to real
units. The internal model reads all three from that directory, so these wrappers
are thin.

| Class | `kind` | Internal model |
|---|---|---|
| `DLinearForecaster` | `dlinear` | `forecasting_evaluation.models.deep_learning_model.dlinear.DLinearModel` |
| `SegRNNForecaster` | `segrnn` | `…segrnn.SegRNNModel` |
| `MixLinearForecaster` | `mixlinear` | `…mixlinear.MixLinearModel` |

All three share the constructor `(model_path, *, normalization_stats_path=None,
device="cuda", **arch)`. Architecture is sourced primarily from the bundled
`training_config.json`; any `arch` keys that name real config fields act as
fallbacks (others are ignored).

### Direct construction

If you don't want to go through `from_release` (e.g. you're pointing at your own
training output), pass the bundle directory directly:

```python
from openmhc.forecasters import DLinearForecaster

fc = DLinearForecaster(
    "releases-fc/openmhc-dlinear-fc/",   # dir with .pypots + training_config.json + scaler
    device="cuda:0",
)
```

The directory **must** contain `training_config.json` (architecture) and the
`standard_scaler_stats.json` (without it, predictions stay in standardized space
instead of real units). This is exactly what the neural bundles ship.

The underlying PyPOTS hyperparameters live in `training_config.json` and mirror
the eval-side configs in
[`forecasting_evaluation/config.py`](../src/forecasting_evaluation/config.py)
(`DLinearModelConfig`, `SegRNNModelConfig`, `MixLinearModelConfig`) — e.g.
`moving_avg_window_size` for DLinear, `seg_len` / `d_model` for SegRNN,
`period_len` / `lpf` / `alpha` / `rank` for MixLinear.

---

## Family B — Foundation fine-tunes

Two wrapper classes around time-series foundation models fine-tuned on the MHC
training split. Both normalize inputs internally, so neither carries a
normalization-stats sidecar (`normalization_stats: null`).

### `Chronos2Forecaster` (`kind: chronos2`)

The released checkpoint is a **merged** (full) HuggingFace model directory, so the
internal `Chronos2Model` loads it directly via `Chronos2Pipeline.from_pretrained`
— no PEFT/LoRA runtime dependency. The MHC fine-tune was LoRA (rank 8, alpha 16)
and merged into the base before publishing.

```python
from openmhc.forecasters import Chronos2Forecaster

fc = Chronos2Forecaster(
    "releases-fc/openmhc-chronos2-fc/checkpoint/",   # merged HF model dir
    device="cuda:0",
    torch_dtype="auto",   # "auto" | "float32" | "float16" | "bfloat16"
)
```

### `TotoForecaster` (`kind: toto`)

The released checkpoint is a PyTorch Lightning `.ckpt`; the internal `TotoModel`
loads it onto the `Datadog/Toto-Open-Base-1.0` backbone (merging LoRA deltas when
present).

```python
from openmhc.forecasters import TotoForecaster

fc = TotoForecaster(
    "releases-fc/openmhc-toto-fc/model.ckpt",
    device="cuda:0",
    context_length=2048,   # tokens of history the backbone attends to
    num_samples=256,       # probabilistic samples drawn per forecast
    lora_alpha=None,       # set only if loading a non-merged LoRA checkpoint
)
```

Both wrappers accept `normalization_stats_path` for the `from_release` contract,
but it is unused (the models normalize internally).

---

## Bundling a release for the leaderboard (`tools/forecasting/build_forecasting_release.py`)

The packager stages all five bundles under `--staging-dir` (default
`releases-fc/`), each with an `openmhc_manifest.json`, the checkpoint payload, and
a Hugging Face model card:

```bash
python tools/forecasting/build_forecasting_release.py \
    --ckpt-root forecasting_model_ckpt \
    --scaler .merge_cache/standard_scaler_stats.json \
    --chronos-merged .merge_cache/chronos2_FT_merged \
    --toto-ckpt /path/to/toto-epoch=24-...-val_loss=-1.3597.ckpt
```

Restrict to specific kinds with repeatable `--only` flags
(`--only dlinear --only chronos2`). Neural bundles can alternatively be staged
straight from a `forecasting_training` release directory (which already carries
its checkpoint, scaler, and rich provenance) via repeatable
`--neural-bundle KIND=DIR`.

Two prep tools produce the inputs that aren't checkpoints-on-disk:

- **`tools/forecasting/merge_chronos_lora.py`** — folds the Chronos-2 LoRA
  adapter into the `amazon/chronos-2` base and re-saves a standalone full model
  directory (`config.json` + `model.safetensors`). Includes an adapter-vs-merged
  self-check that aborts on mismatch.
- **`tools/forecasting/regen_neural_scaler.py`** — deterministically refits the
  train-split `StandardScaler` that the three neural forecasters need to
  inverse-transform predictions. All three share the same scaler, so one fit
  serves all three bundles.

After staging, smoke-test every bundle through the public API:

```bash
python tools/forecasting/verify_bundles.py --staging-dir releases-fc \
  --only dlinear segrnn mixlinear chronos2 toto
```

It loads each bundle with `<Wrapper>.from_release`, runs one synthetic
`predict(history, horizon)`, and checks output shape `(19, horizon)`, finiteness,
and — for neural models — that the co-located scaler was loaded (so predictions
land in raw value space).

---

## Authoring a release manifest by hand

If you trained your own model outside the packaging workflow, write a manifest
directly:

```python
from openmhc.forecasters import write_manifest

write_manifest(
    "releases-fc/my-dlinear/",
    kind="dlinear",
    checkpoint=".",                               # bundle dir holds the .pypots + training_config.json
    arch={"n_steps": 168, "n_pred_steps": 24, "n_features": 19},
    normalization_stats="standard_scaler_stats.json",
    provenance={"trained_by": "my-team", "git_commit": "abc123"},
)
```

Then drop the `.pypots`, `training_config.json`, and `standard_scaler_stats.json`
into that directory. `DLinearForecaster.from_release("releases-fc/my-dlinear/")`
loads it. For a foundation bundle, set `kind="chronos2"` /`"toto"`, point
`checkpoint` at the model directory or `.ckpt`, and leave
`normalization_stats=None`.

---

## Troubleshooting

**`Manifest is for kind 'dlinear', but Chronos2Forecaster expects kind 'chronos2'`** —
You called `from_release` on the wrong wrapper class. Use the wrapper whose
`model_name` matches the manifest's `kind`.

**`ModuleNotFoundError: No module named 'pypots'` / `'chronos'` / `'toto_ts'`** —
Install the optional extra for that family: `pip install 'openmhc[pypots]'`,
`pip install 'openmhc[chronos]'`, or `pip install 'openmhc[toto]'`.

**Neural predictions look standardized (≈0 mean, unit scale) instead of real
units** — The bundle is missing `standard_scaler_stats.json`, or the manifest's
`normalization_stats` is `null`. Neural forecasters need the train-split scaler
to inverse-transform; regenerate it with `tools/forecasting/regen_neural_scaler.py`
and re-stage. `verify_bundles.py` flags this as `scaler_loaded: false`.

**`Merged Chronos-2 model not found … (run tools/forecasting/merge_chronos_lora.py first)`** —
The Chronos-2 bundle needs a *merged* full-model directory (`config.json` +
weights), not the raw LoRA adapter. Run the merge tool to produce it.

**`Unsupported forecasting manifest spec_version …`** — This build understands
`spec_version` 1 only. Re-author or re-stage the bundle.

**Only five wrappers ship.** `openmhc.forecasters` exposes `Chronos2Forecaster`,
`TotoForecaster`, `DLinearForecaster`, `SegRNNForecaster`, `MixLinearForecaster`.
The statistical/naive baselines (`seasonal_naive`, `autoARIMA`, `autoETS`) have
no public wrapper — they are reachable only through the `mhc-forecast-eval` CLI.

---

## Training your own forecaster

The companion package `forecasting_training` ships a Hydra CLI
(`mhc-forecast-train`) for training the three from-scratch PyPOTS forecasters
(DLinear, SegRNN, MixLinear) on the OpenMHC dataset. The output is a
release-bundle directory directly consumable by both the eval CLI's
`model.release_dir=…` flag and this public API's `from_release(...)`.

```bash
mhc-forecast-train \
    model=dlinear \
    seed=42 \
    data.trajectory_hf_dir=/path/to/hourly_trajectory \
    +data.split_file=/path/to/sharable_users.json \
    data.sample_index_file=/path/to/sample_index.json \
    training.epochs=50 \
    output.release_dir=/path/to/my-dlinear-release

# The trained bundle is consumable by mhc-forecast-eval immediately:
mhc-forecast-eval model=dlinear model.release_dir=/path/to/my-dlinear-release ...
```

Swap `model=dlinear` for `segrnn` or `mixlinear`. The YAML schema lives in
[`configs/forecasting_train/`](../configs/forecasting_train/) and the model
factory in
[`src/forecasting_training/model_registry.py`](../src/forecasting_training/model_registry.py).

Fine-tuning the **foundation** models (Chronos-2, Toto) is not part of this CLI —
those checkpoints are produced in the upstream training pipeline, then merged
(Chronos-2) and packaged into release bundles as described above.
