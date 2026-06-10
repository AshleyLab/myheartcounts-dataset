# Paper-results pipelines

Per-track, single-entry orchestrators that turn per-method eval outputs into the
paper leaderboard tables. One subdirectory per track; both follow the same phase
pattern (per-method eval → discover/manifest → skill + rank [→ bootstrap CIs]).

```
scripts/paper_results/
├── forecasting/
│   └── run_paper_pipeline.py     # Track 3: eval -> discover(+validate) -> skill + rank
└── imputation/
    ├── run_paper_pipeline.py     # Track 2: eval -> manifest -> bootstrap -> aggregate
    ├── bootstrap_imputation_draws.py
    └── aggregate_imputation_paper_metrics.py
```

## Forecasting (Track 3)
`forecasting/run_paper_pipeline.py --sweep-config configs/paper/sweep_forecasting.yaml`

- **Phase 0 (eval)** — per *pre-specified* model: `mhc-forecast-eval` under one run
  label (point + binary metrics co-located). `--skip-eval` to reuse existing metrics.
- **Phase 1 (discover + validate)** — select exactly the configured models; **error**
  if any expected model's metrics are missing; ignore extras.
- **Phase 2 (skill + rank)** — `skill_score_summary` + `grouped_metric_rank_summary`
  (continuous=`mae`, binary=`auprc`, vs `seasonal_naive`).
- **Phase 3 (bootstrap CIs + fairness)** — config hook present; not yet implemented.

SLURM one-command: `jobs/simurgh/forecasting_eval/submit_pipeline.sh` (fan out eval
jobs → chain `run_paper_pipeline.sbatch --skip-eval` via `afterok`).

## Imputation (Track 2)
`imputation/run_paper_pipeline.py --sweep-config configs/paper/sweep_methods.yaml`

- **Phase 0 (eval)** — per method: `mhc-impute-eval` with `evaluation.save_pairs=true`.
- **Phase 1 (bootstrap)** — `bootstrap_imputation_draws.py`: paired participant-level
  bootstrap → `bootstrap_draws.parquet`.
- **Phase 2 (aggregate)** — `aggregate_imputation_paper_metrics.py`: skill / rankings /
  fairness with bootstrap CIs → sidecar CSVs.

SLURM: `jobs/sherlock/imputation_eval/submit_all.sh` (fan out → `run_paper_bootstrap.sbatch`
via `afterok`) + `babysit.sh` watchdog.
