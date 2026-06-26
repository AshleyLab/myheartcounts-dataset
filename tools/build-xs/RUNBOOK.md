# OpenMHC-XS → Dataverse — RUNBOOK (tomorrow: just run these)

Goal: build the 593-user XS subset of OpenMHC-Full and upload it to the Harvard
Dataverse draft `doi:10.7910/DVN/LX8Q1G`. Full design + verified file inventory:
see `SPEC.md`. All scripts are in `scripts/` and read shared paths from
`scripts/config.sh`.

```
BUILD = /scratch/users/eggert/openmhc-xs-build
  scripts/   config.sh 00_freeze_check.sh 01_download.sh 02_extract.sh
             make_tiny_subset.py 04_repackage.sh 03_build.sbatch
             05_verify.py 06_upload.sh 07_croissant.sh
  sharable_users_seed42_2026_xs.json     (the 593-user filter key — already here)
  sharable_users_seed42_2026_full.json   (11,894 — shape detection only)
```

## One-time prerequisites (do these before saying "go")
1. **rclone Google auth** (interactive, needs a browser once):
   ```bash
   ml load system rclone
   rclone config         # n) new remote -> name: gdrive -> drive -> scope: drive.readonly
                         # "Use auto config?" -> N ; run `rclone authorize "drive"` on a
                         # laptop, paste the token back. (Folder is Shared-with-me; the
                         # scripts pass --drive-shared-with-me.)
   rclone lsd --drive-shared-with-me gdrive: | grep OpenMHC-Full   # smoke test
   ```
2. **Dataverse API token** (never paste in chat):
   ```bash
   printf '%s' 'YOUR_TOKEN' > ~/.dataverse_token && chmod 600 ~/.dataverse_token
   ```
3. **Confirm the source is frozen** (the GDrive folder is still being edited — e.g.
   `daily_labels_lookup.parquet` was replaced by `daily_labels_lookup_full_history.parquet`
   on 2026-06-23; the scripts handle renamed parquets, but build only on a stable snapshot):
   ```bash
   bash scripts/00_freeze_check.sh      # run twice, a while apart; build only when two snapshots MATCH
   bash scripts/00b_preflight.sh        # peeks inside each archive to confirm tar layout matches 02_extract
   ```

## The run (in order)

| # | Where | Command | ~time |
|---|---|---|---|
| 1 | **DTN** | `bash scripts/01_download.sh` | 1–3 h (38 GB) |
| 2 | **login** | `sbatch scripts/03_build.sbatch` then `squeue --me` | 1–2 h (extract+filter+repackage+verify) |
| 3 | check | `tail -f build.<jobid>.out` ; ensure it ends `OK: all artifacts are subsets...` | — |
| 4 | **DTN** | `bash scripts/06_upload.sh` (dry run first: `DRYRUN=1 bash scripts/06_upload.sh`) | <30 min |
| 5 | **DTN** | `bash scripts/07_croissant.sh` | min |
| 6 | review | open the draft in the Dataverse UI; **publish is manual/your call** | — |

> DTN = `ssh dtn.sherlock.stanford.edu` (has internet; login/compute nodes do not).
> Step 2 runs as one SLURM job so it stays off the login node.

## What each stage does
- **01 download** → `rclone copy` OpenMHC-Full to `$FULL`, verifies with `rclone check`.
- **03 build** (SLURM) → `02_extract.sh` rebuilds the full tree in `$EXTRACT`
  (cat multi-part tars | tar -xz to canonical dirs) → `make_tiny_subset.py`
  filters every artifact to the 593 users into `$STAGE` (writes `_xs_build_report.json`)
  → `04_repackage.sh` tars the 5 big dirs into `$UPLOAD/archives/*_xs.tar.gz` and
  copies the small `data/` files → `05_verify.py` independently re-checks every
  artifact is a subset of the XS split and the inventory is complete.
- **06 upload** → POSTs each file in `$UPLOAD` to the draft (Native API,
  `directoryLabel` preserves structure). Leaves it as a DRAFT.
- **07 croissant** → pulls Dataverse's native Croissant export to
  `openmhc-xs.croissant.json`.

## Filtering reference (baked into make_tiny_subset.py)
- `user_id` is the key everywhere (Arrow/parquet column; JSON dict key).
- copy verbatim: `normalization_stats.json`, `normalization_stats_hourly.json`,
  `task_feature_exclusions.json`, label registries/schemas, READMEs.
- USER_FIRST `{uid:…}`: `enrollment_info.json`, `user_device_info.json`,
  all `forecasting_sample_index/*.json`.
- LABEL_FIRST `{label:{uid:…}}`: `last_labels`, `label_validity`,
  `healthkit_daily`, `clip_dates`, `context_labels` (.json).
- parquet (`df[df.user_id.isin(XS)]`): `labels_wide`, `label_validity`,
  `healthkit_daily`, `daily_labels_lookup`, `weekly_labels_lookup_*`,
  `window_index_w7_s7_d5`.
- HF `load_from_disk→filter→save_to_disk`: `daily_hf`, `daily_hourly_hf`,
  `hourly_trajectory`, `minute_trajectory`.
- hdf5: keep `<uid>.h5`.
- splits: ship `sharable_users_seed42_2026_xs.json` (the XS split) only.
- Unknown/new files: auto-classified by shape, else copied with a warning and
  listed in the report (churn-resilient).

## If something goes wrong
- **Build failed on one artifact** → check `_xs_build_report.json` for the
  `"handler":"ERROR"` entry; re-run just that group (use `python3`, not `python` — the
  module only provides `python3`):
  `python3 scripts/make_tiny_subset.py --full $EXTRACT --out $STAGE --split $SPLIT --only data/labels`.
- **daily_hf extraction too big for SCRATCH headroom** → unlikely (99 TB free),
  but you can point `EXTRACT=$L_SCRATCH/ex` before `sbatch` and add
  `--gres=...`/local-ssd sizing.
- **Upload rejects a large file** → use DVUploader or Dataverse S3 direct-upload
  for that file; the rest of the loop is fine.
- **Croissant 400/not-ready** → fill citation metadata on the draft first, or
  generate locally (mlcroissant / build_croissant.py).

## Open items to confirm before publish (NOT blockers for building)
1. Is `LX8Q1G` the final tiny DOI? (`openmhc/_dataset.py` still hardcodes
   `ZYMJF6`; reconcile ZYMJF6/T7YRIA/XNBITM and update `_VERSION_DOIS`.)
2. Split filename `openmhc.download_dataset(version="tiny")` expects (`_xs` vs `_tiny`).
3. Draft citation metadata / license / authorship — fill in the UI or via API.
