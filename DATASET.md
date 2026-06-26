# OpenMHC Dataset

The MyHeartCounts (MHC) wearable benchmark dataset is hosted separately from this code repo on **Harvard Dataverse**. Two versions are available.

| Version | Size | Use case | DOI |
|---|---|---|---|
| `xs` | ~1.9 GB | Quickstart, NeurIPS reviewer evaluation | [`doi:10.7910/DVN/ZYMJF6`](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZYMJF6) |
| `full` | ~38 GB | Full leaderboard submissions | TBD (released after publication) |

## Download

The Python API wraps the Dataverse access API — no Dataverse account needed for public datasets. Each version must live under its **own** dataset root: the strict resolver cross-checks the root's `dataset_version.json` marker against the `version=` you pass to `evaluate_*`, so xs and full **cannot share a directory**.

```python
import openmhc

# XS subset (recommended first — 593 users, ~1.9 GB)
openmhc.download_dataset(version="xs", dest="~/.cache/openmhc/data-xs")

# Full dataset (when published — 11,894 users, ~38 GB) — distinct root
openmhc.download_dataset(version="full", dest="~/.cache/openmhc/data-full")
```

`download_dataset` writes a `dataset_version.json` marker into `dest` so the eval API can verify it later.

OpenMHC never picks a dataset path implicitly. Pass `dest=` (download) / `data_dir=` (eval) or set `MHC_DATA_DIR` first:

```bash
# Point the eval API at whichever version you want to use
export MHC_DATA_DIR=~/.cache/openmhc/data-xs    # XS for quickstart
export MHC_DATA_DIR=~/.cache/openmhc/data-full  # Full for leaderboard submissions
```

### Restricted access

If the dataset is restricted (DUA-gated), pass your Dataverse API token:

```python
openmhc.download_dataset(version="xs", api_token="<your-token>")
```

Or set the `DATAVERSE_API_TOKEN` environment variable. Get a token at [your Dataverse account page](https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab).

### Manual download

If you'd rather not use the helper (corporate proxies, air-gapped clusters), see [`docs/manual-dataset-setup.md`](docs/manual-dataset-setup.md) for the full per-version recipe (curl + extract + write marker).

## Layout

Each dataset root has the same internal layout. Pick a path per version — e.g. `~/.cache/openmhc/data-xs/` and `~/.cache/openmhc/data-full/` — and point `MHC_DATA_DIR` at whichever you want to use:

```
$MHC_DATA_DIR/
├── dataset_version.json          # marker — version + n_users (required)
├── labels/                       # Track 1
│   ├── last_labels.json          # participant-level outcome labels
│   └── context_labels.json       # participant-level covariates
├── splits/                       # one file, named per version:
│   ├── sharable_users_seed42_2026.json       # full (11,894 users)
│   └── sharable_users_seed42_2026_xs.json    # xs   (   593 users)
├── processed/                    # Tracks 1 + 2
│   ├── daily_hourly_hf/          # daily ×24h sensor tensors (HuggingFace Arrow) — Track 1
│   ├── daily_hf/                 # daily ×1440min sensor tensors (HuggingFace Arrow) — Track 2
│   ├── window_index_w7_s7_d5.parquet         # 7-day weekly window index — Track 1
│   ├── weekly_labels_lookup_stride7_windowed.parquet  # weekly labels lookup (per-task forward window baked in) — Track 1
│   ├── daily_labels_lookup.parquet           # daily labels lookup (windowed; forward-window cap baked in) — Track 1
│   ├── daily_labels_lookup_full_history.parquet  # daily labels lookup (full history; no forward cap; from-raw/xgboost/lsm2) — Track 1
│   └── normalization_stats_hourly.json       # global z-score statistics
├── hourly_trajectory/            # Track 3 — hourly-resolution per-user trajectories
└── forecasting_sample_index/     # Track 3 (P = forecast horizon in hours: 24 or 48)
    ├── sample_index_P_24_M_H_7_3_S_100.json   # used by evaluate_forecasting(forecasting_length=24)
    ├── sample_index_P_48_M_H_7_3_S_100.json   # used by evaluate_forecasting(forecasting_length=48)
    ├── sample_index_P_24_raw.json             # other variants also shipped: _raw, _M, _M_H_7_3
    ├── sample_index_P_48_raw.json
    └── day_remain_mask.json      # per-user retain mask
```

The eval API derives every large payload above from a single dataset root, resolved from an explicit `data_dir=` / `dest=` argument or the `MHC_DATA_DIR` env var. If neither is provided, `openmhc.data_dir()` and the public evaluation APIs raise immediately instead of silently falling back to a default location. If your dataset uses a different layout, the simplest fix is to symlink or rearrange the unpacked files to match.

### The `dataset_version.json` marker

Every dataset root must contain a `dataset_version.json` marker. `download_dataset` writes one automatically; for a hand-assembled root, write one with:

```python
import openmhc
openmhc.write_dataset_marker("~/.cache/openmhc/data-xs", version="xs")     # or "full"
```

The marker pins the version and expected user count so the resolver can fail loudly if a directory ever ends up holding mismatched contents (e.g. an xs split file renamed to the canonical full name). `evaluate_*` reads this marker and cross-checks it against the `version=` you pass, so a wrong root is rejected immediately rather than producing a silently-wrong result.

### Bundled vs. dataset-root files

The schema-only registry files (`label_types.json`, `ordinal_dictionary.json`, `validity_config.json`) ship with this code repo at `data/labels/` and remain the default for bundled metadata. Large JSON label payloads such as `last_labels.json`, `context_labels.json`, `enrollment_info.json`, `label_validity.json`, and `healthkit_daily.json` must come from your explicit dataset root unless you override them with per-file env vars.

## Splits

User-level splits prevent participant leakage between train/validation/test. The canonical split is `sharable_users_seed42_2026.json` — same file used in the paper. Custom splits should be JSON of the form:

```json
{"train": [...], "validation": [...], "test": [...]}
```

with values being participant IDs (strings).

## Label validity

A (user, label) pair is "valid" if the user has sufficient wearable data in the label's time window:

- **C1 / `single_day`** — at least 1 filtered day in the window. Default; maximises participant pool.
- **C2 / `weekly_5of7`** — at least one contiguous 7-day subwindow with ≥ 5 filtered days. Stricter; ~55% smaller.

Submissions to the leaderboard use **C1** by default. Switch with the `data.label_validity_criterion` config.

Validity is **baked into the shipped labels lookups** — a non-sentinel cell in `daily_labels_lookup.parquet` / `weekly_labels_lookup_stride7_windowed.parquet` already marks a valid `(user, day/week)` inside the task window, so the downstream eval reads validity straight from the lookups. (The standalone `label_validity.json` still ships under `labels/` and is consumed separately by the LabelsAPI's `get_labels(return_valid_only=True)`; `label_validity.parquet` is a convenience mirror and `validity_config.json` holds the validity-window thresholds.)

## Data Use Agreement
The MHC dataset is shared under a Data Use Agreement (DUA) covering responsible-use terms for participant-derived health data. Downloading from the Hub triggers a click-through DUA acceptance.
