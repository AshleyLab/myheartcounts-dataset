# SC Cluster Forecasting Evaluation

SLURM wrappers for the public Hydra forecasting CLI (`mhc-forecast-eval`) plus the
one-command **paper pipeline** that produces the full Track-3 leaderboard.

Models: CPU baselines (`seasonal_naive`, `autoARIMA`, `autoETS`) and GPU models
(`chronos2`, `toto`, `dlinear`, `mixlinear`,
`segrnn`). Each `mhc-forecast-eval` run emits point **and** binary metrics
(`auprc/auroc/f1`) co-located under `<model>_metrics/<RUN_LABEL>/`, which the
Layer-2 summaries (skill score + mean rank) read directly.

## Files

**Per-model eval jobs** — each writes `<model>_metrics/<RUN_LABEL>/{point + binary}`:
- `run_naive.sbatch` — `seasonal_naive` (CPU).
- `run_autoets.sbatch` / `run_autoarima.sbatch` — `autoETS` / `autoARIMA` (CPU; autoARIMA resumable).
- `run_chronos2.sbatch` / `run_toto.sbatch` — foundation models, zero-shot + fine-tuned variants (GPU).
- `run_dlinear.sbatch` / `run_mixlinear.sbatch` / `run_segrnn.sbatch` — retrained PyPOTS
  models (GPU; need `MHC_FORECAST_<MODEL>_RELEASE_DIR`).

**Launchers:**
- `submit_pipeline.sh` — **one-command end-to-end**: fan out every eval job under one
  `RUN_LABEL`, then chain the paper pipeline (`afterok`). Writes a job manifest.
- `submit_models.sh` — submit the GPU models under a shared label.
- `submit_all.sh` — submit the CPU baselines under a shared label.

**Layer-2 aggregation:**
- `run_paper_pipeline.sbatch` — runs
  `scripts/paper_results/forecasting/run_paper_pipeline.py --skip-eval`
  (discover + skill + rank); submitted by `submit_pipeline.sh`.
- `skill_rank.sbatch` — standalone skill score + grouped mean-rank for a `RUN_LABEL`.
- `aggregate_results.sbatch` — *deprecated* (MAE-by-channel-hour table only).
- `_common.sh` — conda activation, dataset root, BLAS pinning, output roots.

## Prerequisites (one-time)

The `openmhc` conda env needs the statistical-model deps:

```bash
/simurgh/u/schuetzn/conda/envs/openmhc/bin/pip install sktime pmdarima
```

## Usage

**Full leaderboard, all methods, one command** — fans out every eval job under one
label, then runs the paper pipeline once they finish:

```bash
MHC_FORECAST_RUN_LABEL=forecasting_paper_001 \
  jobs/sc-cluster/forecasting_eval/submit_pipeline.sh
# subset:    jobs/sc-cluster/forecasting_eval/submit_pipeline.sh --only dlinear segrnn
# eval-only: MHC_FORECAST_PIPELINE=0 jobs/sc-cluster/forecasting_eval/submit_pipeline.sh
```
Results land in `results/forecasting_eval/sc-cluster/summary/<RUN_LABEL>/`
(`forecasting_skill_score_*` + `forecasting_grouped_metric_rank_*`).

**CPU baselines only:**

```bash
jobs/sc-cluster/forecasting_eval/submit_all.sh
```

**Single model / re-aggregate an existing label:**

```bash
sbatch jobs/sc-cluster/forecasting_eval/run_autoarima.sbatch
sbatch --export=ALL,MHC_FORECAST_RUN_LABEL=<label> jobs/sc-cluster/forecasting_eval/skill_rank.sbatch
```

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `MHC_REPO_DIR` | Repo checkout | script-resolved repo root |
| `MHC_CONDA_BASE` | conda install prefix | `/simurgh/u/schuetzn/conda` |
| `MHC_CONDA_ENV` | conda env name | `openmhc` |
| `MHC_DATA_DIR` | dataset root (`hourly_trajectory`, `splits`, `forecasting_sample_index`, `labels`) | `/simurgh/u/schuetzn/OpenMHC-Full/data` |
| `MHC_FORECAST_RUNS_ROOT` | output root | `results/forecasting_eval/sc-cluster` |
| `MHC_FORECAST_RUN_LABEL` | shared run label | `forecasting_<jobid>` |
| `MHC_FORECAST_OVERWRITE` | overwrite existing parquets (else resume by skipping) | `false` |
| `MHC_FORECAST_AGGREGATE` | (legacy `submit_all`/`submit_models`) set `0` to skip the deprecated aggregation | `1` |
| `MHC_FORECAST_PIPELINE` | (`submit_pipeline.sh`) set `0` to submit eval jobs only, no paper pipeline | `1` |
| `MHC_FORECAST_<MODEL>_RELEASE_DIR` | release bundle for a trained model (`DLINEAR`/`SEGRNN`/`MIXLINEAR`/`CHRONOS2`/`TOTO`) | `submit_pipeline.sh` defaults |

## Notes

- Account/partition: `--account=simurgh --partition=simurgh` (CPU-only — no
  `--gres` requested). Core counts are kept modest: the harness loop is
  sequential, so only `autoARIMA` (and marginally `autoETS`) benefit from cores.
- BLAS threads are pinned to 1 in `_common.sh` so joblib (`n_jobs=-1`) does not
  oversubscribe.
- Each run persists raw `(ground_truth, prediction)` pairs plus a `scales.npz`
  under its `*_metrics/<RUN_LABEL>/` dir, enabling post-hoc metric recomputation
  and bootstrapping without re-running models.
- `autoARIMA` is resumable: re-submitting continues from where it stopped
  (existing per-user parquets are skipped while `MHC_FORECAST_OVERWRITE=false`).
