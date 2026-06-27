# Paper-results pipelines

Per-track, single-entry orchestrators that turn per-method eval outputs into the
paper leaderboard tables. One subdirectory per track; all follow the same phase
pattern (per-method eval → discover/manifest/bootstrap → skill + rank [+ fairness] [→ CIs]).

```
scripts/paper_results/
├── downstream/                  # Track 1: eval -> bootstrap -> skill + rank + fairness
│   ├── run_paper_pipeline.py
│   ├── bootstrap_downstream_draws.py
│   ├── aggregate_downstream_paper_metrics.py
│   ├── aggregate_fairness_skill_score.py
│   ├── leaderboard/             # substrate producer + build_leaderboard_json.py
│   └── dev/                     # repro/parity gate + appendix metrics (not run-path)
├── forecasting/
│   └── run_paper_pipeline.py     # Track 3: eval -> discover(+validate) -> skill + rank
└── imputation/
    ├── run_paper_pipeline.py     # Track 2: eval -> manifest -> bootstrap -> aggregate
    ├── bootstrap_imputation_draws.py
    └── aggregate_imputation_paper_metrics.py
```

## Downstream (Track 1)
`downstream/run_paper_pipeline.py --config configs/paper/downstream_paper.yaml`

Config-driven (a provenance YAML, not a `--sweep-config`): one file records the methods,
bootstrap count/seed, baseline, and fairness knobs.

- **Phase 0 (eval)** — per method: `run_eval.py` (the same `evaluate_prediction` call an
  external submitter makes) → `eval_<m>.csv` + per-(method, task) predictions. `--skip-eval`
  to reuse frozen predictions (the published numbers come from logged SLURM eval jobs).
- **Phase 1 (bootstrap)** — `bootstrap_downstream_draws.py`: paired user-level bootstrap
  → `bootstrap_draws.parquet`.
- **Phase 2 (aggregate)** — `aggregate_downstream_paper_metrics.py` (skill / rank / fairness)
  + `aggregate_fairness_skill_score.py` (disparity-ratio Fairness Skill Score + BCa) →
  sidecar CSVs; `build_leaderboard_json.py` renders the leaderboard.

## Forecasting (Track 3)
`forecasting/run_paper_pipeline.py --sweep-config configs/paper/sweep_forecasting.yaml`

- **Phase 0 (eval)** — per *pre-specified* model: `mhc-forecast-eval` under one run
  label (point + binary metrics co-located). `--skip-eval` to reuse existing metrics.
- **Phase 1 (discover + validate)** — select exactly the configured models; **error**
  if any expected model's metrics are missing; ignore extras.
- **Phase 2 (skill + rank)** — `skill_score_summary` + `grouped_metric_rank_summary`
  (continuous=`mae`, binary=`auprc`, vs `seasonal_naive`).
- **Phase 3 (bootstrap CIs + fairness)** — config hook present; not yet implemented.

SLURM one-command: `jobs/sc-cluster/forecasting_eval/submit_pipeline.sh` (fan out eval
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
