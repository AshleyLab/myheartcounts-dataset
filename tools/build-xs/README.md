# OpenMHC-XS build pipeline

Provenance + reproduction code for **OpenMHC-XS**, the 5% (593-user) subset of the
full OpenMHC dataset published on Harvard Dataverse at
[`doi:10.7910/DVN/ZYMJF6`](https://doi.org/10.7910/DVN/ZYMJF6).

This is kept on a **separate branch** (not merged to `main`) purely as a record of
how the subset was generated; it is not part of the `openmhc` package.

## What it does

Takes the full OpenMHC release (hosted in the Google Drive folder *OpenMHC-Full*),
filters every artifact down to the 593 users in
`sharable_users_seed42_2026_xs.json` (train 356 / val 59 / test 178; a clean ~5%
subset of the 11,894-user full split, seed 42), repackages the archives with an
`_xs` suffix, verifies, and uploads to Dataverse.

## Inputs

- **Full dataset:** Google Drive *OpenMHC-Full* (downloaded via `rclone`, read-only).
- **Subset definition:** `sharable_users_seed42_2026_xs.json` — the canonical XS split,
  shipped in this repo at `data/splits/` and in the published Dataverse bundle (ZYMJF6).
- **Dataverse API token:** read from `~/.dataverse_token` (chmod 600) — never committed.

## Pipeline (see `RUNBOOK.md` + `SPEC.md` for full detail)

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `scripts/config.sh` | shared env (paths, DOI, token file) — sourced by the others |
| 0 | `scripts/00_freeze_check.sh`, `scripts/00b_preflight.sh` | confirm the GDrive source is stable / tar layout is as expected |
| 1 | `scripts/01_download.sh` | `rclone` the full bundle from GDrive |
| 2 | `scripts/02_extract.sh` | unpack the full archives |
| 3 | `scripts/make_tiny_subset.py` (via `scripts/03_build.sbatch`) | filter every artifact to the 593 XS users (pyarrow at shard level) |
| 4 | `scripts/04_repackage.sh` | re-tar the filtered tree as `*_xs.tar.gz` |
| 5 | `scripts/05_verify.py` | assert each artifact is a non-empty subset of 593 users |
| 6 | `scripts/06_upload.sh` | upload to Dataverse via the Native API (idempotent; DOI-overridable) |
| 7 | `scripts/07_croissant.sh` | emit Croissant JSON-LD metadata (optional; post-publish) |

## Notes

- HF Arrow datasets are filtered with **pyarrow** at the shard level (no `datasets`
  dependency) so any consumer version can still load the result.
- Heavy work (`03_build.sbatch`) runs as a Slurm job on Sherlock; the download/upload
  steps run from an internet-connected node (DTN / code-server).
