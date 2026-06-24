# Sherlock imputation-eval runner

End-to-end SLURM batch for running every openmhc imputer (`mhc-impute-eval`)
on Stanford's Sherlock cluster, with strict mask parity to the
`MHC-benchmark` `max91d` ablation that is the headline number in the paper.

## Layout

```
jobs/sherlock/imputation_eval/
├── _common.sh               # sourced by every sbatch (env + venv + paths)
├── 00_setup.sh              # one-time: venv check + W&B checkpoint downloads
├── run_baselines.sbatch     # CPU: mean, mode, linear, locf,
│                            #      temporal_*, personalized_* (9 methods)
├── run_brits.sbatch         # GPU: any
├── run_dlinear.sbatch       # GPU: any  (smallest, 24h, 64G)
├── run_dlinear_weekly.sbatch      # GPU: 7-day DLinear
├── run_fedformer.sbatch     # GPU: any
├── run_timesnet.sbatch      # GPU: GPU_MEM:24GB
├── run_lsm2.sbatch          # GPU: GPU_MEM:40GB
├── run_lsm2_weekly.sbatch         # GPU: GPU_MEM:32GB — dense 7-day LSM-2
├── run_lsm2_weekly_sparse.sbatch  # GPU: GPU_MEM:32GB, --mem=96G, batch=16
├── run_lsm2_weekly_sparse_24gb.sbatch  # alt: --mem=128G, batch=8 (host-OOM mitigation)
├── sweep_methods.yaml       # 17-method paper sweep (incl. lsm2_weekly dense)
├── sweep_methods_no_dense.yaml    # 16-method sweep (excl. lsm2_weekly) — bootstrap parity
├── run_paper_bootstrap.sbatch     # Phase 1 + 2 on 17 methods (writes paper/)
├── run_paper_bootstrap_no_dense.sbatch  # Phase 1 + 2 on 16 methods (writes paper_no_dense/)
├── upload_leaderboard.sbatch      # Phase D: push substrate + bootstrap to HF
├── submit_all.sh            # MASTER LAUNCHER — submits per-method eval jobs
├── babysit.sh               # /loop driver
├── _babysit.py              # actual watchdog (called by babysit.sh)
├── verify_parity.py         # compares new results vs MHC-benchmark max91d
└── README.md                # this file
```

## Deliverables (per method)

Every imputer job writes to `${RUNS_ROOT}/<method>/` where
`RUNS_ROOT=${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/runs` (defaults to
`/scratch/users/$USER/...` via `jobs/sherlock/_env.sh`). That dir contains:

| file | purpose |
|------|---------|
| `results.json` | point-estimate metrics per (scenario, split, channel-group) |
| `bootstrap_metrics.json` | participant-level cluster bootstrap CIs (n_boot=1000) |
| `pairs/` | per-channel ground-truth/prediction Parquet (re-bootstrappable) |
| `.hydra/config.yaml` | full resolved Hydra config (provenance) |
| `openmhc_manifest.json` | provenance: W&B artifact id, val_mae, etc. |

After Phase 1 + 2 of the paper bootstrap, `${PAPER_OUT}/` contains:

| file | purpose |
|------|---------|
| `bootstrap_draws.parquet` | shared-draw error matrix, re-aggregatable |
| `skill_scores_bootstrap.csv` | macro skill score vs LOCF + 95% CI |
| `avg_rankings_bootstrap.csv` | average rank across methods + 95% CI |
| `fairness_skill_score_bootstrap.csv` | disparity-ratio Fairness Skill Score (leaderboard's `fair_skill_score`) + 95% CI |
| `bootstrap_method_dirs.json` | provenance: which pairs/ fed each method |

## Prerequisites

1. **openmhc venv** somewhere on `$SCRATCH` (created with `python -m venv`
   against `/share/software/user/open/python/3.12.1`). Has openmhc, pypots,
   pytorch_lightning, hydra-submitit-launcher. `scripts/dev/activate-openmhc.sh`
   activates it.
2. **dataset cache** under `${MHC_CACHE}` (defaults to
   `${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-full/`, populated by
   `python -c "import openmhc; openmhc.download_dataset(version='full')"` — 292
   GB, 11894 users).
3. **W&B credentials** at `~/.netrc` (`machine api.wandb.ai`) — needed for
   downloading paper checkpoints.

## End-to-end usage

```bash
cd "$REPO"   # your clone of myheartcounts-dataset

# (one-time) verify env + download all 6 paper checkpoints
bash jobs/sherlock/imputation_eval/00_setup.sh

# (optional) interactive smoke test before submitting
sh_dev -t 1:00:00
source jobs/sherlock/imputation_eval/_common.sh
mhc-impute-eval method=mean data=xs masking=sleep_gap_only \
  hydra.run.dir=$OUT_BASE/_smoke \
  data.daily_hf_dir=${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-xs/processed/daily_hf \
  data.split_file=${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-xs/splits/sharable_users_seed42_2026.json
ls $OUT_BASE/_smoke/{results.json,per_user_errors.parquet,pairs}

# submit everything (7 imputer jobs + 1 paper-bootstrap with afterok dep)
bash jobs/sherlock/imputation_eval/submit_all.sh

# kick off the hourly babysitter (resubmits failed jobs up to 3 times)
/loop 1h bash jobs/sherlock/imputation_eval/babysit.sh

# once jobs complete, check parity vs MHC-benchmark max91d
python jobs/sherlock/imputation_eval/verify_parity.py
```

## Resource summary

| Group | Partition | Time | CPUs | Mem | GPU | Constraint |
|-------|-----------|------|------|-----|-----|-----------|
| baselines (9 methods) | `normal` | 24h | 12 | 128G | — | — |
| BRITS | `gpu` | 48h | 8 | 96G | 1 | any |
| DLinear | `gpu` | 24h | 8 | 64G | 1 | any |
| FEDformer | `gpu` | 48h | 8 | 96G | 1 | any |
| TimesNet | `gpu` | 48h | 8 | 96G | 1 | `GPU_MEM:24GB` |
| LSM2 daily | `gpu` | 48h | 8 | 96G | 1 | `GPU_MEM:40GB` |
| LSM2 weekly sparse | `gpu` | 48h | 8 | 96G | 1 | `GPU_MEM:32GB` |
| LSM2 weekly sparse 24GB | `gpu` | 48h | 8 | 128G | 1 | — (any GPU_MEM) |
| LSM2 weekly dense | `gpu` | 48h | 8 | 96G | 1 | `GPU_MEM:32GB` |
| paper-bootstrap (×2) | `normal` | 8h | 12 | 128G | — | — |
| substrate producer (array) | `normal` | 45m | 8 | 32G | — | — |
| HF leaderboard upload | `normal` | 1h | 2 | 8G | — | — |

Mirrors `MHC-benchmark/jobs/stanford/sherlock/imputation/eval/*.sbatch`.

The paper-bootstrap walltime ceiling of 8h is a conservative cap — the actual
Phase 1+2 run on the full 16/17-method set takes ~1–2h.

## Re-bootstrap without re-eval

The cross-imputer bootstrap is fully reproducible from the saved `pairs/`
dirs. To re-run just Phase 1 + 2 (e.g., to try a different baseline):

```bash
source jobs/sherlock/imputation_eval/_common.sh
# edit sweep_methods.yaml: change `baseline_method`, `n_boot`, etc.
python scripts/paper_results/imputation/run_paper_pipeline.py \
  --sweep-config jobs/sherlock/imputation_eval/sweep_methods.yaml \
  --skip-eval
```

To re-aggregate from an existing `bootstrap_draws.parquet` (Phase 2 only):

```bash
python scripts/paper_results/imputation/run_paper_pipeline.py \
  --sweep-config jobs/sherlock/imputation_eval/sweep_methods.yaml \
  --skip-eval --skip-phase1
```

## Babysitter behaviour

`babysit.sh` (a thin wrapper around `_babysit.py`) reads
`${OUT_BASE}/job_manifest.tsv`, queries `sacct` for the latest jobid per
label, and:

- leaves PENDING / RUNNING alone;
- treats COMPLETED + expected output present as **done**;
- if COMPLETED but artifacts missing → counts as failure;
- on FAILED / TIMEOUT / OUT_OF_MEMORY / NODE_FAIL / CANCELLED →
  resubmits the same sbatch, appends a new manifest row;
- when an imputer is resubmitted while paper_bootstrap is queued, it
  rewrites the dependency in place via
  `scontrol update jobid=... Dependency=afterok:...`;
- caps resubmissions at **3 per label**. Beyond that, the row is logged
  as `STUCK` and skipped — manual intervention needed.

Triggered hourly via:

```
/loop 1h bash "$REPO/jobs/sherlock/imputation_eval/babysit.sh"
```

A one-line summary of every action is appended to
`${OUT_BASE}/babysit.log`.

## Parity verification

`verify_parity.py` compares each new run's headline metrics against the
corresponding MHC-benchmark max91d run:

- continuous: `mean_normalized_rmse` / `mean_normalized_mae`, 1% relative tolerance
- binary: `macro_balanced_accuracy` / `macro_roc_auc`, 0.005 absolute tolerance

Exit code 0 = all checked rows pass, 1 = at least one fails. Run after
the imputer jobs complete; mismatches → inspect:

1. `${RUNS_ROOT}/<method>/.hydra/config.yaml` vs the MHC-benchmark
   `config.yaml` (look for `n_days`, `inference_batch_size`,
   `normalization_*` divergences);
2. `${RUNS_ROOT}/<method>/openmhc_manifest.json::provenance.wandb_artifact`
   — confirm the same checkpoint version was used;
3. masks: both should read from
   `data/imputation/masks/sharable_users_seed42_2026_max91d`;
4. normalization stats: the release bundle's
   `normalization_stats.json` must match the one MHC-benchmark trained
   against (`data/processed/pypotsdaily_h5/4bb0aa42/normalization_stats.json`).

## Refreshing the HF leaderboard (canonical recipe)

This is how the current leaderboard parquets at
[`MyHeartCounts/OpenMHC-leaderboard-data`](https://huggingface.co/datasets/MyHeartCounts/OpenMHC-leaderboard-data)
under `imputation/` were produced. Four phases — A is the heavy lift, B+C+D
are a ~3-hour dependency chain.

### Phase A — per-method evaluation

Submit every imputer (8 sbatch jobs; the `baselines` job sequentially runs
all 9 CPU baselines):

```bash
bash jobs/sherlock/imputation_eval/submit_all.sh --no-paper
# --no-paper because phases B–D below will run the bootstrap with the
# 17-method sweep, including lsm2_weekly which submit_all.sh doesn't cover.
```

`submit_all.sh` does **not** include `run_lsm2_weekly.sbatch` (the dense
weekly variant is "out of the main paper table" and not in the JOBS map).
Run it once on its own when needed:

```bash
sbatch jobs/sherlock/imputation_eval/run_lsm2_weekly.sbatch
```

After Phase A: 17 directories under `${RUNS_ROOT}/<method>/` each with
`pairs/`, `per_user_errors.parquet` (subgroup_attr=all only), `results.json`,
and `openmhc_manifest.json`. Wallclock: ~1 day end-to-end (mixed CPU+GPU).

### Phase B — paper-bootstrap aggregator (×2 in parallel)

Run two paper-bootstrap variants in parallel so consumers can pick either the
backward-compatible 16-method pool (rank-matched to the prior leaderboard) or
the 17-method pool (includes the new dense weekly LSM-2):

```bash
JID_A=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap.sbatch)
JID_B=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap_no_dense.sbatch)
```

Outputs:

- 17-method → `${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper/`
- 16-method → `${SCRATCH_RUN_ROOT}/openmhc-imputation-eval/paper_no_dense/`

Each dir gets `bootstrap_draws.parquet` (~110 MB), `bootstrap_method_dirs.json`,
and the Phase-2 CSVs (`skill_scores_bootstrap.csv`,
`avg_rankings_bootstrap.csv`, `fairness_skill_score_bootstrap.csv`). Wallclock:
~1–2h, parallel.

### Phase C — substrate producer (array, 17 tasks)

`runs/<m>/per_user_errors.parquet` (Phase A) has only `subgroup_attr=all`.
The HF substrate format expects subgroup-broken-down rows for `age_group` and
`sex` so the leaderboard can recompute fairness-skill-score. The producer
takes the existing `pairs/` + demographics lookup and writes the expanded
parquets:

```bash
JID_C=$(sbatch --parsable --dependency=afterok:$JID_A \
  scripts/paper_results/imputation/parity/produce_per_method_per_user_errors.sbatch)
```

Depends on **only** `JID_A` (it needs `paper/bootstrap_method_dirs.json` to
resolve pairs paths — `JID_B`'s manifest in `paper_no_dense/` would also
work). Outputs 17 parquets at `paper-verification/per_user/<method>.parquet`
(~150K rows each). Wallclock: ~45 min, parallel across array tasks.

### Phase D — HF upload

```bash
JID_D=$(sbatch --parsable --dependency=afterok:$JID_B:$JID_C \
  jobs/sherlock/imputation_eval/upload_leaderboard.sbatch)
```

Uploads everything to `MyHeartCounts/OpenMHC-leaderboard-data`:

| Local source | HF destination |
|---|---|
| `paper-verification/per_user/<method>.parquet` (×17) | `imputation/<method>.parquet` (16 refreshed + 1 new for `lsm2_weekly`) |
| `paper_no_dense/bootstrap_draws.parquet` | `imputation/bootstrap/draws.parquet` (**replaces** canonical) |
| `paper/bootstrap_draws.parquet` | `imputation/bootstrap_with_dense_weekly/draws.parquet` (new sibling) |

Each method's `.meta.json` sidecar is written on every run (every upload
passes `--results-json`, which auto-extracts `fallback_rate`), but the upload
tool fetches the existing HF sidecar first and merges in only the fields it was
given. So the display metadata (`display_name`/`type`/`submitter`/`subtrack`)
of the 16 already-on-HF methods is preserved verbatim; only `lsm2_weekly` ships
fresh display fields, on top of its auto-extracted `fallback_rate`.

**Auth gotcha**: `huggingface_hub` 1.4.1 resolves `HF_HOME` to
`/tmp/huggingface` by default and never finds the cached login token at
`~/.cache/huggingface/token`. `upload_leaderboard.sbatch` works around this
by reading the token file and exporting `HF_TOKEN` before any HF API call.
If you re-run from an interactive shell, do the same:

```bash
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
```

### One-button rebuild

```bash
# Phase A (~1 day wallclock):
bash jobs/sherlock/imputation_eval/submit_all.sh --no-paper
sbatch jobs/sherlock/imputation_eval/run_lsm2_weekly.sbatch

# Wait for everything in `squeue -u $USER` to clear, then verify with the
# fallback-rate snippet under `scripts/paper_results/imputation/` (or just
# `ls ${RUNS_ROOT}/*/results.json`).

# Phases B + C + D (~3 h wallclock, fully chained):
JID_A=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap.sbatch)
JID_B=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap_no_dense.sbatch)
JID_C=$(sbatch --parsable --dependency=afterok:$JID_A \
         scripts/paper_results/imputation/parity/produce_per_method_per_user_errors.sbatch)
JID_D=$(sbatch --parsable --dependency=afterok:$JID_B:$JID_C \
         jobs/sherlock/imputation_eval/upload_leaderboard.sbatch)
echo "Chain: A=$JID_A B=$JID_B C=$JID_C D=$JID_D"
```

### Adding a new method to the leaderboard

1. Register the method via either a Hydra method preset under
   `configs/imputation/method/<name>.yaml` or an entry in
   `src/imputation_evaluation/hydra/registry.py` (Phase A needs this; the
   later phases just consume pairs/ by name).
2. Add a sbatch under this directory; wire it into `submit_all.sh`'s `JOBS`
   map if you want the babysitter to track it.
3. Add the method to both `sweep_methods.yaml` and
   `sweep_methods_no_dense.yaml` (the latter only if you want it in the
   16-method legacy pool too).
4. Bump `--array=0-N` in
   `scripts/paper_results/imputation/parity/produce_per_method_per_user_errors.sbatch`
   and append the method name to its `METHODS` array.
5. Add the method to `METHODS` in `upload_leaderboard.sbatch`. If it's new
   on HF, pass `--name/--type/--submitter/--subtrack` so the display sidecar
   is created with the right strings on the first upload.
   `upload_leaderboard_substrate.py` is called with `--results-json
   "${RUNS}/${M}/results.json"`, so the `fallback_rate` sidecar field is
   auto-extracted (worst-case `overall_fallback_rate` across all
   `(scenario, split)` cells) and merged into the existing sidecar without
   clobbering the display fields. See
   `tools/leaderboard_docs/imputation/SCHEMA.md` for the sidecar schema.
6. Re-run the one-button rebuild.

## Known open items

- **`daily_hf_dir`**: defaults to
  `${MHC_CACHE}/processed/daily_hf` (which expands to
  `${SCRATCH_RUN_ROOT}/.myheartcounts-dataset-cache/data-full/processed/daily_hf`).
  Override `MHC_CACHE` or `SCRATCH_RUN_ROOT` if your layout differs.
- **Personalized baselines**: added stub YAMLs at
  `configs/imputation/method/personalized_{mean,mode,temporal_mean}.yaml`
  to match the existing pattern. Smoke-test before trusting full runs.
- **W&B credentials**: 00_setup.sh assumes `~/.netrc` is configured.
  If not, set `WANDB_API_KEY` in your shell before running.
