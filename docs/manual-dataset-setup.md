# Manual Dataset Setup

Use this guide if you cannot use `openmhc.download_dataset()` — for example, on air-gapped clusters, behind a corporate proxy, or when you already have the raw Dataverse bundle on disk.

Both dataset versions share the same final layout under one explicit root directory. Set `MHC_DATA_DIR` first or substitute your chosen path directly in the commands below. `~/.cache/openmhc/data` is still a valid choice, but it is no longer implicit.

---

## XS version (~1.9 GB, 593 users)

**Source:** Dataverse deposit `doi:10.7910/DVN/ZYMJF6`

### Step 1 — Download the bundle

```bash
# Public dataset — no token needed
curl -L -o openmhc-xs.zip \
  "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/ZYMJF6"

# Restricted dataset — supply your Dataverse API token
curl -L -o openmhc-xs.zip \
  -H "X-Dataverse-key: <your-token>" \
  "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/ZYMJF6"
```

### Step 2 — Extract the outer ZIP

```bash
CACHE="${MHC_DATA_DIR:-$HOME/.cache/openmhc/data}"
mkdir -p "$CACHE"
unzip openmhc-xs.zip -d "$CACHE"
```

The ZIP contains an `archives/` directory with five tar.gz files and a `data/` tree with metadata.

### Step 3 — Copy metadata into the cache root

```bash
cp -r "$CACHE/data/splits"                    "$CACHE/"
cp -r "$CACHE/data/labels"                    "$CACHE/"
cp -r "$CACHE/data/forecasting_sample_index"  "$CACHE/"
mkdir -p "$CACHE/processed"
cp    "$CACHE/data/processed/"*.json          "$CACHE/processed/"
cp    "$CACHE/data/processed/"*.parquet       "$CACHE/processed/"
```

### Step 4 — Extract the HuggingFace Arrow datasets

The XS archives unpack with an `_xs` suffix on the top-level directory that must be stripped.

```bash
# Daily 1440-min tensors (Track 2 — imputation)
tar -xzf "$CACHE/archives/daily_hf_xs.tar.gz" -C "$CACHE/processed/"
mv "$CACHE/processed/daily_hf_xs" "$CACHE/processed/daily_hf"

# Daily 24-hour tensors (Track 1 — outcome prediction)
tar -xzf "$CACHE/archives/daily_hourly_hf_xs.tar.gz" -C "$CACHE/processed/"
mv "$CACHE/processed/daily_hourly_hf_xs" "$CACHE/processed/daily_hourly_hf"

# Hourly per-user trajectories (Track 3 — forecasting)
tar -xzf "$CACHE/archives/hourly_trajectory_xs.tar.gz" -C "$CACHE/"
mv "$CACHE/hourly_trajectory_xs" "$CACHE/hourly_trajectory"

# Per-minute per-user trajectories (Track 3 — optional high-res input)
tar -xzf "$CACHE/archives/minute_trajectory_xs.tar.gz" -C "$CACHE/"
mv "$CACHE/minute_trajectory_xs" "$CACHE/minute_trajectory"
```

### Step 5 — Extract the HDF5 archive

```bash
mkdir -p "$CACHE/hdf5"
tar -xzf "$CACHE/archives/hdf5_sharable_2026_xs.tar.gz" -C "$CACHE/hdf5/"
```

### Step 6 — Move normalization stats

If `normalization_stats.json` landed at the cache root rather than in `processed/`:

```bash
[ -f "$CACHE/normalization_stats.json" ] && \
  mv "$CACHE/normalization_stats.json" "$CACHE/processed/"
```

---

## Full version (~38 GB, 11,894 users)

**Source:** Dataverse deposit `doi:10.7910/DVN/XNBITM` (released after publication)

The full bundle splits large archives into numbered parts (`*.tar.gz.part-NN`) that must be concatenated before extraction. All directories extract to canonical names — no renaming required.

### Step 1 — Download the bundle

```bash
curl -L -o openmhc-full.zip \
  -H "X-Dataverse-key: <your-token>" \
  "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/XNBITM"
```

### Step 2 — Extract the outer ZIP

```bash
CACHE="${MHC_DATA_DIR:-$HOME/.cache/openmhc/data}"
mkdir -p "$CACHE"
unzip openmhc-full.zip -d "$CACHE"
```

### Step 3 — Copy metadata into the cache root

```bash
cp -r "$CACHE/data/splits"                    "$CACHE/"
cp -r "$CACHE/data/labels"                    "$CACHE/"
cp -r "$CACHE/data/forecasting_sample_index"  "$CACHE/"
mkdir -p "$CACHE/processed"
cp    "$CACHE/data/processed/"*.json          "$CACHE/processed/"
cp    "$CACHE/data/processed/"*.parquet       "$CACHE/processed/"
```

### Step 4 — Extract multi-part archives

Each `cat ... | tar -xz` command concatenates the numbered parts and streams them directly into tar — no intermediate file needed.

```bash
# Daily 1440-min tensors — 5 parts → processed/daily_hf/
cat "$CACHE/archives/daily_hf_full.tar.gz.part-"* | tar -xz -C "$CACHE/processed/"

# Daily 24-hour tensors — single archive → processed/daily_hourly_hf/
tar -xzf "$CACHE/archives/daily_hourly_hf_full.tar.gz" -C "$CACHE/processed/"

# HDF5 archives — 5 parts → hdf5/
mkdir -p "$CACHE/hdf5"
cat "$CACHE/archives/hdf5_sharable_2026_full.tar.gz.part-"* | tar -xz -C "$CACHE/hdf5/"

# Hourly trajectories — single archive → hourly_trajectory/
tar -xzf "$CACHE/archives/hourly_trajectory_full.tar.gz" -C "$CACHE/"

# Per-minute trajectories — 10 parts → minute_trajectory/
cat "$CACHE/archives/minute_trajectory_full.tar.gz.part-"* | tar -xz -C "$CACHE/"
```

---

## Expected layout after setup

After either version, `$MHC_DATA_DIR` should look like:

```
$MHC_DATA_DIR/
├── labels/
│   ├── last_labels.json
│   ├── context_labels.json
│   ├── label_validity.json
│   ├── clip_dates.json
│   └── ...
├── splits/
│   ├── sharable_users_seed42_2026_xs.json    # XS split (593 users)
│   └── sharable_users_seed42_2026.json       # Full split (11,894 users)
├── processed/
│   ├── daily_hf/                             # Track 2 — 1440-min tensors (HuggingFace Arrow)
│   ├── daily_hourly_hf/                      # Track 1 — 24-hour tensors (HuggingFace Arrow)
│   ├── normalization_stats.json
│   ├── normalization_stats_hourly.json
│   ├── window_index_w7_s7_d5.parquet
│   ├── daily_labels_lookup.parquet
│   └── weekly_labels_lookup_stride7.parquet
├── forecasting_sample_index/                 # Track 3
│   ├── sample_index_raw.json
│   └── ...
├── hourly_trajectory/                        # Track 3
├── minute_trajectory/                        # Track 3
└── hdf5/                                     # HDF5 copies of sensor data
```

## Pointing the API at your data directory

```bash
export MHC_DATA_DIR=/path/to/your/data
```

Or pass it explicitly:

```python
import openmhc
results = openmhc.evaluate_imputation(MyImputer(), data_dir="/path/to/your/data")
```

`openmhc.data_dir()` returns the explicit root resolved from `data_dir=` or `MHC_DATA_DIR`, so you can verify the lookup:

```python
import openmhc
print(openmhc.data_dir())   # → /path/to/your/data
```
