# Forecasting track — leaderboard substrate

This directory holds the **per-method substrate** for the OpenMHC Track-3
(forecasting) leaderboard. Each method ships two files:

```
forecasting/<method>.parquet         # per-(user × channel × metric) raw values
forecasting/<method>.meta.json       # display + diagnostic sidecar
```

See `SCHEMA.md` for the exact column / field schema (including the
`fallback_rate` diagnostic).

## What's it for

The substrate parquets are the canonical inputs for:

- The OpenMHC HF Space (`MyHeartCounts/OpenMHC`) — it downloads these parquets +
  the sidecars and runs the canonical forecasting reducers to produce the live
  leaderboard table (skill / fair-skill / mean-rank vs `seasonal_naive`).
- Independent re-aggregation (skill / rank / fairness reducers in
  `src/forecasting_evaluation/metrics/`).
- The cluster-bootstrap reference at `forecasting/bootstrap/` (per-draw CIs) is
  reduced from these substrates, so any change here must be matched by a
  bootstrap refresh (see "Refreshing" below) or the CIs drift off the points.

## Loading

```python
from huggingface_hub import hf_hub_download
import pandas as pd, json

parquet = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "forecasting/chronos2_zeroshot.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(parquet)
print(df.shape, df.columns.tolist())

# Display + diagnostic sidecar (incl. fallback_rate)
meta_p = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "forecasting/chronos2_zeroshot.meta.json",
    repo_type="dataset",
)
print(json.loads(open(meta_p).read()))
# -> {"display_name": "Chronos-2 (zero-shot)", "type": "Foundation Model", ...,
#     "fallback_rate": 0.0}
```

## Pooled substrate

The pooled per-user frame across all methods is the concatenation of the
per-method parquets (~7,370 rows/method × 10 methods for the canonical config):

```python
import glob, pandas as pd
pooled = pd.concat(
    [pd.read_parquet(p) for p in glob.glob("forecasting/*.parquet")],
    ignore_index=True,
)
```

## `fallback_rate` is on by default

The invalid-prediction rate is **always** produced and **always** threaded into
the sidecar — there is no opt-in flag:

- The eval harness always records `overall_fallback_rate` (fraction of forecast
  cells the model emitted as NaN, which the harness substituted with
  Seasonal-Naive before scoring) at the top level of each run's `results.json`,
  and `evaluate_forecasting` carries it in the substrate parquet's `meta`.
- `stage_leaderboard_substrates.py` emits each upload command **with**
  `--results-json <runs>/<method>/hydra/results.json`, so
  `upload_leaderboard_substrate.py` auto-extracts the rate and writes the
  `fallback_rate` sidecar key by default. Existing display fields are preserved.

A `fallback_rate > ~5%` means the headline scores are inflated with baseline
performance on the substituted cells and should be read with caution.

## Refreshing (full paper-results reproduction)

**Single source of truth — avoid regressing to an old run.** The canonical run
is pinned in exactly one place: `run_label` / `output_root` in
`configs/paper/sweep_forecasting.yaml` (currently `forecasting_full_20260622`).
The substrate-staging and bootstrap-draws scripts **default to it** (they read
`output_root` from that file), so a bare `stage_leaderboard_substrates.py` or
`produce_forecasting_bootstrap_draws.py` cannot silently rebuild the leaderboard
from a stale substrate. To re-point the canonical run, edit **only** the sweep
config. Likewise the **methodology is fixed in the sweep + code**:
`within_user_aggregation: micro` (binary AUROC is **pooled per user** over all
the user's horizon cells — the eval emits one pooled row/user; the legacy
per-window "macro" path is *not* used for the leaderboard). All steps run from
the OpenMHC code repo on Simurgh (SC); see
`jobs/sc-cluster/forecasting_eval/README.md` for cluster details.

```bash
LABEL=forecasting_full_20260622

# (1) Eval + aggregate — fan out all 10 model jobs under one label, then chain
#     the paper pipeline (substrate + skill/rank + bootstrap CIs + fairness).
#     Each run writes results.json with the top-level overall_fallback_rate.
MHC_FORECAST_RUN_LABEL=$LABEL jobs/sc-cluster/forecasting_eval/submit_pipeline.sh
#   re-aggregate only (metrics already on disk):
#   sbatch --export=ALL,MHC_FORECAST_RUN_LABEL=$LABEL \
#     jobs/sc-cluster/forecasting_eval/run_paper_pipeline.sbatch
# -> results/forecasting_eval/simurgh/summary/$LABEL/
#      {forecasting_per_user_errors.parquet, skill_rank_models.json,
#       forecasting_skill_score*.csv, forecasting_grouped_metric_rank*.csv,
#       forecasting_fairness_skill_score*.csv}

# (2) Bootstrap-draws reference for the leaderboard CIs (n_boot=1000; CPU ~30 min).
sbatch scripts/paper_results/forecasting/produce_forecasting_bootstrap_draws.sbatch \
    --summary-dir results/forecasting_eval/simurgh/summary/$LABEL
# -> $LABEL/bootstrap_draws.parquet (+ .meta.json)

# (3) Stage per-method substrates + emit the upload commands (each already
#     carries --results-json so fallback_rate is auto-filled).
python scripts/paper_results/forecasting/stage_leaderboard_substrates.py
#   ^ prints one `upload_leaderboard_substrate.py ... --track forecasting
#     --results-json <runs>/<method>/hydra/results.json` per method; run them
#     (HF auth required: HF_TOKEN or `huggingface-cli login`). This writes
#     forecasting/<method>.{parquet,meta.json} with fallback_rate.

# (4) Upload the bootstrap-draws reference (overwrites forecasting/bootstrap/).
python tools/upload_leaderboard_bootstrap.py \
    --dir results/forecasting_eval/simurgh/summary/$LABEL --track forecasting

# (5) Docs — keep this README + SCHEMA.md in sync on the dataset repo:
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for src, dst in [
    ("tools/leaderboard_docs/forecasting/SCHEMA.md", "forecasting/SCHEMA.md"),
    ("tools/leaderboard_docs/forecasting/README.md", "forecasting/README.md"),
]:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                    repo_id="MyHeartCounts/OpenMHC-leaderboard-data", repo_type="dataset",
                    commit_message=f"docs(forecasting): sync {dst}")
PY
```

Steps (2) and (4) keep the bootstrap CIs on the same canonical substrate as the
point numbers — always run them together when the substrates change.
