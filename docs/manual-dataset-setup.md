# Manual Dataset Setup

Use this guide if you cannot use `openmhc.download_dataset()` — for example, on air-gapped clusters, behind a corporate proxy, or when you already have the raw Dataverse bundle on disk.

**One root per version.** Every OpenMHC dataset root must contain a `dataset_version.json` marker declaring which release lives there (`xs` or `full`), and the public eval API cross-checks that marker against the `version=` you pass. The two versions therefore cannot share a directory — put each under its own root and point `MHC_DATA_DIR` (or `data_dir=`) at whichever one you want to use.

Recommended layout for keeping both versions side by side:

```
~/.cache/openmhc/
├── data-xs/      # 593-user reviewer subset
└── data-full/    # 11,894-user paper release
```

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
CACHE="$HOME/.cache/openmhc/data-xs"
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

The XS archives unpack straight to their target directory names (`daily_hf`,
`daily_hourly_hf`, `hourly_trajectory`, `minute_trajectory`), so no renaming is
needed.

```bash
# Daily 1440-min tensors (Track 2 — imputation)
tar -xzf "$CACHE/archives/daily_hf_xs.tar.gz" -C "$CACHE/processed/"

# Daily 24-hour tensors (Track 1 — outcome prediction)
tar -xzf "$CACHE/archives/daily_hourly_hf_xs.tar.gz" -C "$CACHE/processed/"

# Hourly per-user trajectories (Track 3 — forecasting)
tar -xzf "$CACHE/archives/hourly_trajectory_xs.tar.gz" -C "$CACHE/"

# Per-minute per-user trajectories (Track 3 — optional high-res input)
tar -xzf "$CACHE/archives/minute_trajectory_xs.tar.gz" -C "$CACHE/"
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

### Step 7 — Write the dataset version marker

```bash
python -c "import openmhc; openmhc.write_dataset_marker('$CACHE', version='xs')"
```

This drops `dataset_version.json` at the root. Without it the eval API will refuse to resolve this root.

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
CACHE="$HOME/.cache/openmhc/data-full"
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

# Per-minute trajectories — 10 parts → minute_trajectory/  (optional;
# not used by the public evaluate_* API — safe to skip if disk is tight)
cat "$CACHE/archives/minute_trajectory_full.tar.gz.part-"* | tar -xz -C "$CACHE/"
```

Extracted footprint is roughly 100–500 GB depending on whether you include `minute_trajectory/`. Only `daily_hf/`, `daily_hourly_hf/`, and `hourly_trajectory/` are read by the public eval API; the others mirror the canonical Dataverse layout.

### Step 5 — Write the dataset version marker

```bash
python -c "import openmhc; openmhc.write_dataset_marker('$CACHE', version='full')"
```

If you want the marker to also pin the row count of `processed/daily_hf` (cheap extra check the resolver will validate):

```bash
python -c "
import openmhc
from datasets import load_from_disk
root = '$CACHE'
n = len(load_from_disk(f'{root}/processed/daily_hf'))
openmhc.write_dataset_marker(root, version='full', daily_hf_rows=n)
"
```

---

## Expected layout after setup

After either version, the chosen cache root should look like:

```
$CACHE/
├── dataset_version.json                       # marker — REQUIRED
├── labels/
│   ├── last_labels.json
│   ├── context_labels.json
│   ├── label_validity.json
│   ├── clip_dates.json
│   └── ...
├── splits/
│   └── sharable_users_seed42_2026.json        # full root
│       (or sharable_users_seed42_2026_xs.json  for xs root)
├── processed/
│   ├── daily_hf/                              # Track 2 — 1440-min tensors (HuggingFace Arrow)
│   ├── daily_hourly_hf/                       # Track 1 — 24-hour tensors (HuggingFace Arrow)
│   ├── normalization_stats.json
│   ├── normalization_stats_hourly.json
│   ├── window_index_w7_s7_d5.parquet
│   ├── daily_labels_lookup.parquet
│   ├── daily_labels_lookup_full_history.parquet
│   └── weekly_labels_lookup_stride7_windowed.parquet
├── forecasting_sample_index/                  # Track 3
│   ├── sample_index_P_24_M_H_7_3_S_100.json   # used by evaluate_forecasting (horizon 24)
│   ├── day_remain_mask.json
│   └── ...
├── hourly_trajectory/                         # Track 3
├── minute_trajectory/                         # Track 3 — optional
└── hdf5/                                      # HDF5 copies of sensor data
```

## Pointing the API at your data directory

Point `MHC_DATA_DIR` at the version you want to run against:

```bash
export MHC_DATA_DIR=~/.cache/openmhc/data-xs    # for xs eval / quickstart
export MHC_DATA_DIR=~/.cache/openmhc/data-full  # for full / leaderboard
```

Or pass it explicitly per call (and remember to pass `version=`, which the API now requires):

```python
import openmhc
results = openmhc.evaluate_imputation(
    MyImputer(),
    version="full",
    data_dir="~/.cache/openmhc/data-full",
)
```

`openmhc.data_dir()` returns the explicit root resolved from `data_dir=` or `MHC_DATA_DIR`, so you can verify the lookup:

```python
import openmhc
print(openmhc.data_dir())                                # → /path/to/your/data
print(openmhc.read_dataset_marker(openmhc.data_dir()))   # → {'version': 'full', ...}
```
