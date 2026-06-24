# Imputation track — leaderboard substrate

This directory holds the **per-method substrate** for the OpenMHC Track-2
(imputation) leaderboard. Each method ships two files:

```
imputation/<method>.parquet         # per-(user × scenario × channel × subgroup) errors
imputation/<method>.meta.json       # display + diagnostic sidecar
```

See `SCHEMA.md` for the exact column / field schema (including the
`fallback_rate` diagnostic).

## What's it for

The substrate parquets are the canonical inputs for:

- The OpenMHC HF Space (`MyHeartCounts/OpenMHC`) — `leaderboard_compute.py`
  there downloads these parquets + the sidecars and runs the canonical
  reducers from `imputation_evaluation` to produce the live leaderboard
  table.
- Independent re-aggregation (skill / rank / fairness reducers in
  `src/imputation_evaluation/evaluation/`)
- The cluster-bootstrap reference at `imputation/bootstrap*/` is reduced
  from these substrates, so any change here propagates downstream.

## Loading

```python
from huggingface_hub import hf_hub_download
import pandas as pd, json

# One method's substrate
parquet = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "imputation/locf.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(parquet)
print(df.shape, df.columns.tolist())

# Display + diagnostic sidecar (incl. fallback_rate)
meta_p = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "imputation/locf.meta.json",
    repo_type="dataset",
)
meta = json.loads(open(meta_p).read())
print(meta)
# -> {"display_name": "LOCF (baseline)", "type": "Statistical", ...,
#     "fallback_rate": 0.0}
```

## Pooled substrate (BCa LOO)

The pooled per-user errors frame across all methods is NOT stored here — it
is exactly the concatenation of the per-method parquets:

```python
import glob, pandas as pd
pooled = pd.concat(
    [pd.read_parquet(p) for p in glob.glob("imputation/*.parquet")],
    ignore_index=True,
)
# ~2.5M rows = 148,510 rows/method × 17 methods (or × 16 for the legacy pool)
```

The bootstrap reference under `imputation/bootstrap/` was computed against
the **16-method** pool (legacy / paper-matching). The sibling
`imputation/bootstrap_with_dense_weekly/` was computed against the
**17-method** pool that includes the `lsm2_weekly` dense variant.

## Refreshing

The substrate is produced and uploaded by the OpenMHC code repo:

```bash
# (Phase A) Per-method runs land at runs/<method>/{pairs/, per_user_errors.parquet, results.json}
bash jobs/sherlock/imputation_eval/submit_all.sh --no-paper

# (Phase B+C+D) Bootstrap → substrate producer → HF upload (chained)
JID_A=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap.sbatch)
JID_B=$(sbatch --parsable jobs/sherlock/imputation_eval/run_paper_bootstrap_no_dense.sbatch)
JID_C=$(sbatch --parsable --dependency=afterok:$JID_A \
         scripts/paper_results/imputation/parity/produce_per_method_per_user_errors.sbatch)
JID_D=$(sbatch --parsable --dependency=afterok:$JID_B:$JID_C \
         jobs/sherlock/imputation_eval/upload_leaderboard.sbatch)
```

The upload step auto-extracts `fallback_rate` from each method's
`results.json` and threads it into the sidecar without clobbering the
existing display fields. See
`jobs/sherlock/imputation_eval/README.md` for the canonical recipe.
