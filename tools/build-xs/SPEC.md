# OpenMHC-XS → Harvard Dataverse: build & upload spec

**Status:** SPEC ONLY — do not execute yet (GDrive source is actively changing — see §0).
**Date:** 2026-06-22 (inventory + schemas verified against the live GDrive folder and on-disk data)
**Owner repo:** `myheartcounts-dataset` (`openmhc`). MHC-benchmark/DVC are NOT used as the build location; the old export scripts are referenced only as ground-truth for filter logic.

---

## 0. ⚠️ Source is in flux — verify before building

The GDrive folder is being edited **today**: `data/processed/weekly_labels_lookup_stride7_windowed.parquet` was uploaded 2026-06-22 19:01 by `yuzebai03@gmail.com` (a rename/replacement of the previous `weekly_labels_lookup_stride7.parquet`), and `daily_labels_lookup.parquet` was re-uploaded the same minute. **Do not build until the owners confirm the folder is frozen.** The build script must first snapshot a `manifest.json` (id, size, md5, modifiedTime per file) and refuse to run if it drifts.

---

## 1. Goal

Produce **OpenMHC-XS** — the existing 5% subset (593 users) of the full OpenMHC release — and publish it to the Harvard Dataverse draft **`doi:10.7910/DVN/LX8Q1G`**, mirroring the full release's artifact set 1:1. Optionally generate Croissant metadata and upload programmatically.

Pipeline: **rclone download (DTN) → filter to 593 users (SLURM) → repackage → Croissant → Native-API upload (DTN) → verify + wire DOI into openmhc.**

---

## 2. The XS user set (the filter key) — VERIFIED

- `sharable_users_seed42_2026_xs.json` — **593 users** (train 356 / val 59 / test 178), 4.99% of 11,894, verified clean subset. Source: `git show origin/main:data/splits/sharable_users_seed42_2026_xs.json` in `myheartcounts-dataset`. Saved to `/scratch/users/eggert/openmhc-xs-build/`.
- **The user-id key is `user_id` everywhere it's stored** (Arrow column, every parquet column, and the dict key in the JSONs). The Labels API arg is named `health_code` and the labels README calls it `healthCode`, but the *stored* name is `user_id` / the bare id string. Filter on the literal `user_id`.
- **ID format is mixed** in the namespace: both 24-char health-codes (e.g. `-72lErpUJHq6Yom2XiuN57mJ`) and 36-char UUIDs (e.g. `000e397b-…`) appear in the full split. The XS set is a subset of those exact strings, so exact-string membership works — **but** note `context_labels.json` keys are the UUID form specifically. Validation must confirm matched-user counts per artifact (don't assume 593 everywhere).

---

## 3. COMPLETE GDrive inventory (OpenMHC-Full, verified) + per-file action

Folder id `1YH8BozsH5VW_9MUx8UEIWxPULJ1Wl-zB`, owner schuetzn@stanford.edu.

### Top level (3 files + 2 folders)
| File | Size | Action |
|---|---|---|
| `README.md` | 4.7 KB | regenerate for XS |
| `normalization_stats.json` | 677 B | **copy whole** (global minute-res z-score stats) |
| `task_feature_exclusions.json` | 1.8 KB | **copy whole** (downstream-eval config) |
| `archives/` | — | see below |
| `data/` | — | see below |

### `archives/` — 22 files, ~38 GB compressed (the big per-user data)
Each is filtered by loading the dataset, filtering on `user_id`, re-saving, re-tarring.
| Archive (parts) | Extracts to | ~size | Filter | Old script? |
|---|---|---|---|---|
| `hdf5_sharable_2026_full.tar.gz.part-00..04` | `data/hdf5/<user_id>.h5` | 8.9 GiB | keep the 593 `<user_id>.h5` files | ✅ `export_hdf5_subset.py` |
| `daily_hf_full.tar.gz.part-00..04` | `data/processed/daily_hf/` | 8.0 GiB | HF `load_from_disk`→`.filter(user_id∈XS)`→`save_to_disk` (851 shards, **one row per user-day**) | ✅ `export_hf_subset.py` |
| `daily_hourly_hf_full.tar.gz` | `data/processed/daily_hourly_hf/` | 0.95 GB | same HF filter (only 5,941 users have it → expect ≤593) | ❌ NEW |
| `hourly_trajectory_full.tar.gz` | `data/hourly_trajectory/` | 1.2 GB | same HF filter (one row per user → 593 rows) | ❌ NEW |
| `minute_trajectory_full.tar.gz.part-00..09` | `data/minute_trajectory/` | 18.7 GiB | same HF filter (one row per user → 593 rows) | ❌ NEW |

### `data/splits/` — 1 file
| File | Action |
|---|---|
| `sharable_users_seed42_2026.json` (514 KB, full split only — **no XS here**) | **replace**: ship `sharable_users_seed42_2026_xs.json` (the 593 set from the repo) as the canonical split |

### `data/labels/` — 16 files
| File | Size | Structure | Action |
|---|---|---|---|
| `enrollment_info.json` | 1.75 MB | `{user_id: {...}}` | **USER_FIRST** — filter outer keys |
| `user_device_info.json` | 3.8 MB | `{user_id: {phone,watch}}` | **USER_FIRST** — filter outer keys |
| `last_labels.json` | 108 MB | `{label: {user_id: {...}}}` | **LABEL_FIRST** — filter inner keys |
| `label_validity.json` | 30 MB | `{label: {user_id: [...]}}` | **LABEL_FIRST** — filter inner keys |
| `healthkit_daily.json` | 376 MB | `{label: {user_id: {...}}}` | **LABEL_FIRST** — filter inner keys (big → stream or ≥a few GB RAM) |
| `clip_dates.json` | 21 MB | `{label: {user_id: "date"}}` | **LABEL_FIRST** — filter inner keys |
| `context_labels.json` | 43 MB | `{label: {user_id(UUID): {...}}}` | **LABEL_FIRST** — filter inner keys (UUID ids) |
| `labels_wide.parquet` | 842 KB | 11,894 rows × 171 cols, `user_id` | **PARQUET** — `df[df.user_id.isin(XS)]` |
| `label_validity.parquet` | 358 KB | 11,894 × 41, `user_id` | **PARQUET** |
| `healthkit_daily.parquet` | 23 MB | 6.77M rows, long `(user_id,label,date,value)` | **PARQUET** |
| `label_types.json` | 13 KB | `{label: {...}}` (169) | **copy whole** (registry) |
| `ordinal_dictionary.json` | 6.6 KB | `{label: {...}}` (36) | **copy whole** |
| `label_pretty_names.json` | 7.5 KB | `{label: {...}}` | **copy whole** |
| `validity_config.json` | 1.1 KB | `{label: int}` (41) | **copy whole** |
| `README.md` | 4.2 KB | doc | **copy whole** |
| `RELEASE_NOTES.md` | 15.6 KB | doc | **copy whole** |

### `data/processed/` — 4 files
| File | Size | Action | Old script? |
|---|---|---|---|
| `daily_labels_lookup.parquet` | 10.4 MB | **PARQUET** filter `user_id` | ✅ |
| `weekly_labels_lookup_stride7_windowed.parquet` | 3.45 MB | **PARQUET** filter `user_id` (⚠ renamed/new today, §0) | ⚠ (was `…_stride7.parquet`) |
| `window_index_w7_s7_d5.parquet` | 4.7 MB | **PARQUET** filter `user_id` | ❌ NEW |
| `normalization_stats_hourly.json` | 672 B | **copy whole** (global hourly z-score stats) | n/a |

### `data/forecasting_sample_index/` — 9 files (all USER_FIRST, ❌ none in old scripts)
`{user_id: [int day-indices]}` → filter outer keys by `user_id`:
`sample_index_P_24_raw.json`, `sample_index_P_24_M.json`, `sample_index_P_24_M_H_7_3.json`, `sample_index_P_24_M_H_7_3_S_100.json`, `sample_index_P_48_raw.json`, `sample_index_P_48_M.json`, `sample_index_P_48_M_H_7_3.json`, `sample_index_P_48_M_H_7_3_S_100.json`, `day_remain_mask.json`.

---

## 4. Filter logic — ground truth (from the old export scripts)

All scripts pool users across all split keys: `users = {u for v in split.values() for u in v}`.
Four reusable handlers (port into a new `make_tiny_subset.py` in `openmhc`):
- **USER_FIRST** `{uid: v}` → `{uid: v for uid in users}`
- **LABEL_FIRST** `{label: {uid: v}}` → `{label: {uid: v for uid in inner if uid in users}}`
- **PARQUET** → `df[df["user_id"].isin(users)]`
- **HF dataset** → `ds.filter(lambda b: [u in users for u in b["user_id"]], batched=True)` → `save_to_disk`
- **per-user files** (hdf5) → keep `<uid>.h5`
- **copy whole** → registries, schemas, global stats, docs

**Gaps to implement net-new** (old scripts don't cover): `daily_hourly_hf`, `hourly_trajectory`, `minute_trajectory` (HF filter), `window_index_w7_s7_d5.parquet` (PARQUET), all 9 `forecasting_sample_index/*.json` (USER_FIRST).

**Baked decisions:** normalization stats (minute + hourly) ship **as-is** (canonical, computed on full train split — do NOT recompute); ship the XS split; both `.json` and `.parquet` variants of healthkit_daily / label_validity must be filtered and kept consistent.

---

## 5. Prerequisites (need from user)

1. **rclone Google auth (headless):** `ml load system rclone` → `rclone config` new remote `gdrive` type `drive` scope `drive.readonly`; answer N to auto-config, run `rclone authorize "drive"` on a browser machine, paste token. Folder is *Shared with me* → use `--drive-shared-with-me` or set `root_folder_id = 1YH8BozsH5VW_9MUx8UEIWxPULJ1Wl-zB`.
2. **Dataverse API token:** `printf '%s' 'TOKEN' > ~/.dataverse_token && chmod 600 ~/.dataverse_token`.

---

## 6. Phase 1 — Download (DTN: `dtn.sherlock.stanford.edu`)
Compute nodes have no internet — transfer on the DTN.
```bash
ml load system rclone
DEST=$SCRATCH/openmhc-xs-build/full
rclone copy --drive-shared-with-me -P --transfers 4 --checkers 8 "gdrive:OpenMHC-Full" "$DEST"
rclone check --drive-shared-with-me "gdrive:OpenMHC-Full" "$DEST" --one-way   # vs manifest.json
# reconstruct multi-part archives: cat <name>.part-* | tar -xz -C <target>
```

## 7. Phase 2 — Filter (SLURM `normal`, ~8 CPU / 64 GB / 4 h)
Driver `make_tiny_subset.py`. Extract big archives into **`$L_SCRATCH`** (node-local SSD, auto-wiped) so the full uncompressed data never lands on Lustre; write only the 593-user outputs to `$SCRATCH`. Confirm `daily_hf` uncompressed size on first extract (can be large; stream shard-by-shard if needed). After each artifact, assert matched-user count is plausible (≤593; daily_hourly_hf < 593).

## 8. Phase 3 — Repackage
Rebuild the `OpenMHC-Full` tree as `OpenMHC-XS/` with identical paths, 593 users. Re-tar the per-user/HF datasets (`*_xs.tar.gz`). XS total ~1–2 GB → **single tarballs, no multi-part splitting** (the full release split only to beat upload limits). Regenerate top-level `README.md` (XS counts, LX8Q1G DOI, install).

## 9. Phase 4 — Croissant
- **A (recommended):** Dataverse native exporter after files+metadata exist: `GET …/api/datasets/export?exporter=croissant&persistentId=doi:10.7910/DVN/LX8Q1G` (with token). Zero authoring; NeurIPS-accepted.
- **B (richer RAI):** `mlcroissant`, or adapt MHC-benchmark `croissant_metadata_and_rai/build_croissant.py` (enhances the Dataverse auto-export to Croissant 1.1 + RAI).

## 10. Phase 5 — Upload (Native API; DTN)
```bash
TOKEN=$(cat ~/.dataverse_token); PID="doi:10.7910/DVN/LX8Q1G"
BASE="https://dataverse.harvard.edu/api/datasets/:persistentId"
curl -s -H "X-Dataverse-key:$TOKEN" "$BASE/?persistentId=$PID" | python3 -m json.tool | head   # confirm draft writable
# per file (directoryLabel preserves subfolder structure):
curl -s -H "X-Dataverse-key:$TOKEN" -X POST \
  -F "file=@OpenMHC-XS/archives/daily_hf_xs.tar.gz" \
  -F 'jsonData={"description":"daily_hf (5% XS)","directoryLabel":"archives"}' \
  "$BASE/add?persistentId=$PID"
```
Prefer **pyDataverse** / **DVUploader** for the S3 direct-upload flow on larger files. Fill/verify citation metadata first. **Leave as draft** — publishing is the user's call (`POST $BASE/actions/:publish?...&type=major`).

## 11. Phase 6 — Verify & wire up
- Assert each artifact's user set ⊆ XS and counts are as expected; fail loudly otherwise.
- **DOI wiring:** `openmhc/src/openmhc/_dataset.py` hardcodes tiny=`doi:10.7910/DVN/ZYMJF6`; target is `LX8Q1G`. Reconcile the three stray DOIs (ZYMJF6 / T7YRIA / XNBITM) and update `_VERSION_DOIS`. Smoke-test `openmhc.download_dataset(version="tiny")`.

## 12. Open questions / risks
1. **GDrive not frozen** (§0) — weekly lookup renamed/added today; re-verify before any run.
2. **Canonical tiny DOI** — is LX8Q1G final (vs ZYMJF6 which already resolves)?
3. **daily_hf uncompressed size** — confirm on first extract (may force stream-filter).
4. **Split filename openmhc expects** — `_xs.json` vs `_tiny.json`.
5. **Draft metadata/license/authorship** — confirm or fill with the token.
6. **Mixed id formats / context_labels UUIDs** — validate per-artifact matched counts.

## 13. Rough effort
Download ~1–3 h · filter+repackage ~1–2 h (1 SLURM job) · upload <30 min · Croissant minutes · wire+verify ~30 min. ~half a day once auth+token are in place and GDrive is frozen.
