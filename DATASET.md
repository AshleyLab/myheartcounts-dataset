# OpenMHC Dataset

The MyHeartCounts (MHC) wearable benchmark dataset is hosted separately from this code repo on **Harvard Dataverse**. Two versions are available.

| Version | Size | Use case | DOI |
|---|---|---|---|
| `tiny` | ~TBD MB | Quickstart, NeurIPS reviewer evaluation | [`doi:10.7910/DVN/ZYMJF6`](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZYMJF6) |
| `full` | ~TBD GB | Full leaderboard submissions | TBD (released after publication) |

## Download

The Python API wraps the Dataverse access API — no Dataverse account needed for public datasets:

```python
import openmhc

# Tiny (recommended first)
openmhc.download_dataset(version="tiny")

# Full (when published)
openmhc.download_dataset(version="full")
```

By default the data is cached at `~/.cache/openmhc/data`. Override with the `dest=` argument or the `MHC_DATA_DIR` environment variable.

```bash
export MHC_DATA_DIR=/path/to/your/data
```

### Restricted access

If the dataset is restricted (DUA-gated), pass your Dataverse API token:

```python
openmhc.download_dataset(version="tiny", api_token="<your-token>")
```

Or set the `DATAVERSE_API_TOKEN` environment variable. Get a token at [your Dataverse account page](https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab).

### Manual download

If you'd rather not use the helper (corporate proxies, air-gapped clusters), use `curl`:

```bash
# Tiny
curl -L -o openmhc-tiny.zip \
  "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/ZYMJF6"
unzip openmhc-tiny.zip -d ~/.cache/openmhc/data
```

For restricted datasets, add `-H "X-Dataverse-key: <your-token>"`.

## Layout

After download, `$MHC_DATA_DIR` should look like:

```
$MHC_DATA_DIR/
├── labels/                       # Track 1
│   ├── last_labels.json          # participant-level outcome labels
│   ├── context_labels.json       # participant-level covariates
│   ├── label_validity.json       # which (user, label) pairs pass validity
│   └── clip_dates.json           # per-task date clipping (longitudinal labels — optional)
├── splits/
│   └── sharable_users_seed42_2026.json   # canonical user-level splits
├── processed/                    # Tracks 1 + 2
│   ├── daily_hourly_hf/          # daily ×24h sensor tensors (HuggingFace Arrow) — Track 1
│   ├── daily_hf/                 # daily ×1440min sensor tensors (HuggingFace Arrow) — Track 2
│   ├── window_index_w7_s7_d5.parquet         # 7-day weekly window index — Track 1
│   ├── weekly_labels_lookup_stride7.parquet  # weekly labels lookup — Track 1
│   ├── daily_labels_lookup.parquet           # daily labels lookup — Track 1
│   └── normalization_stats_hourly.json       # global z-score statistics
├── hourly_trajectory/            # Track 3 — hourly-resolution per-user trajectories
└── forecasting_sample_index/     # Track 3
    ├── sample_index_raw.json     # forecast-window sample index
    ├── sample_index_P_48_raw.json
    ├── sample_index_MH_7_3S_100.json
    ├── sample_index_P_48_M_H_7_3_S_100.json
    └── day_remain_mask.json      # per-user retain mask
```

The eval API resolves these paths through ``openmhc._evaluate._DatasetPaths`` — every entry above is derived from a single root (``MHC_DATA_DIR`` / explicit ``data_dir=`` arg / ``~/.cache/openmhc/data``). If your dataset uses a different layout, the simplest fix is to symlink or rearrange the unpacked files to match.

The schema-only registry files (`label_types.json`, `ordinal_dictionary.json`, `validity_config.json`) ship with this code repo at `data/labels/` and don't need to be downloaded.

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

## Data Use Agreement

The MHC dataset is shared under a Data Use Agreement (DUA) covering responsible-use terms for participant-derived health data. Downloading from the Hub triggers a click-through DUA acceptance. By using the dataset you agree to:

- Use the data for academic research only
- Not attempt to re-identify participants
- Not redistribute the raw data
- Cite the paper

Full DUA text: <https://myheartcounts.stanford.edu/dua>
