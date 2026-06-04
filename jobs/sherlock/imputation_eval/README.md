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
├── run_fedformer.sbatch     # GPU: any
├── run_timesnet.sbatch      # GPU: GPU_MEM:24GB
├── run_lsm2.sbatch          # GPU: GPU_MEM:40GB
├── run_lsm2_weekly_sparse.sbatch  # GPU: GPU_MEM:40GB
├── sweep_methods.yaml       # Sherlock overlay of configs/paper/sweep_methods.yaml
├── run_paper_bootstrap.sbatch     # Phase 1 + 2 (depends on all imputer jobs)
├── submit_all.sh            # MASTER LAUNCHER — submits everything
├── babysit.sh               # /loop driver
├── _babysit.py              # actual watchdog (called by babysit.sh)
├── verify_parity.py         # compares new results vs MHC-benchmark max91d
└── README.md                # this file
```

## Deliverables (per method)

Every imputer job writes to `${RUNS_ROOT}/<method>/` where
`RUNS_ROOT=/scratch/users/schuetzn/openmhc-imputation-eval/runs`. That dir
contains:

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
| `fairness_subgroup_scores_bootstrap.csv` | per-(age_group, sex) skill scores + CIs |
| `bootstrap_method_dirs.json` | provenance: which pairs/ fed each method |

## Prerequisites

1. **openmhc venv** at `/scratch/users/schuetzn/envs/openmhc/` (already
   exists — created with `python -m venv` against
   `/share/software/user/open/python/3.12.1`). Has openmhc, pypots,
   pytorch_lightning, hydra-submitit-launcher.
2. **dataset cache** at
   `/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-full/`
   (already populated, 292 GB, 11894 users).
3. **W&B credentials** at `~/.netrc` (`machine api.wandb.ai`) — needed for
   downloading paper checkpoints.

## End-to-end usage

```bash
cd /home/users/schuetzn/myheartcounts-dataset

# (one-time) verify env + download all 6 paper checkpoints
bash jobs/sherlock/imputation_eval/00_setup.sh

# (optional) interactive smoke test before submitting
sh_dev -t 1:00:00
source jobs/sherlock/imputation_eval/_common.sh
mhc-impute-eval method=mean data=xs masking=sleep_gap_only \
  hydra.run.dir=$OUT_BASE/_smoke bootstrap=on \
  data.daily_hf_dir=/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-xs/processed/daily_hf \
  data.split_file=/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-xs/splits/sharable_users_seed42_2026.json
ls $OUT_BASE/_smoke/{results.json,bootstrap_metrics.json,pairs}

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
| LSM2 weekly sparse | `gpu` | 48h | 8 | 96G | 1 | `GPU_MEM:40GB` |
| paper-bootstrap | `normal` | 8h | 12 | 128G | — | — |

Mirrors `MHC-benchmark/jobs/stanford/sherlock/imputation/eval/*.sbatch`.

## Re-bootstrap without re-eval

The cross-imputer bootstrap is fully reproducible from the saved `pairs/`
dirs. To re-run just Phase 1 + 2 (e.g., to try a different baseline):

```bash
source jobs/sherlock/imputation_eval/_common.sh
# edit sweep_methods.yaml: change `baseline_method`, `n_boot`, etc.
python scripts/paper_results/run_paper_pipeline.py \
  --sweep-config jobs/sherlock/imputation_eval/sweep_methods.yaml \
  --skip-eval
```

To re-aggregate from an existing `bootstrap_draws.parquet` (Phase 2 only):

```bash
python scripts/paper_results/run_paper_pipeline.py \
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
/loop 1h bash /home/users/schuetzn/myheartcounts-dataset/jobs/sherlock/imputation_eval/babysit.sh
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

## Known open items

- **`daily_hf_dir`**: confirmed at
  `/scratch/users/schuetzn/.myheartcounts-dataset-cache/data-full/processed/daily_hf`.
  If you point at a different cache, edit `_common.sh`.
- **Personalized baselines**: added stub YAMLs at
  `configs/imputation/method/personalized_{mean,mode,temporal_mean}.yaml`
  to match the existing pattern. Smoke-test before trusting full runs.
- **W&B credentials**: 00_setup.sh assumes `~/.netrc` is configured.
  If not, set `WANDB_API_KEY` in your shell before running.
