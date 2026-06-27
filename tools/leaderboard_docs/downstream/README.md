# Downstream track — leaderboard substrate

This directory holds the **per-method substrate** for the OpenMHC Track-1 (outcome
prediction) leaderboard. Each method ships two files:

```
downstream/<method>.parquet         # per-user prediction pairs (method × task × subgroup × user)
downstream/<method>.meta.json       # display + diagnostic sidecar
```

See `SCHEMA.md` for the exact column / field schema (including the
`fallback_rate` diagnostic), and `bootstrap/` for the per-draw CI reference.

Unlike Tracks 2/3 (per-user MAE), Track-1's headline metrics — binary AUPRC, ordinal
Spearman, regression Pearson — are cohort-level ranking / correlation metrics that do
not decompose into one error per user. The substrate therefore ships the raw per-user
`(y_true, y_pred, y_proba)` pairs, and the leaderboard recomputes the paired metrics
server-side against the `linear` baseline.

## What's it for

The substrate parquets are the canonical inputs for:

- The OpenMHC HF Space (`MyHeartCounts/OpenMHC`) — the live leaderboard table (skill /
  fair-skill / mean-rank vs the `linear` baseline). Track-1's headline scores are
  paired-bootstrap means, too heavy to reduce on each page load, so the maintainers reduce
  these substrates offline and publish the per-method rows as `downstream/leaderboard_rows.json`,
  which the Space reads directly.
- Independent re-aggregation (the reducers in `scripts/paper_results/downstream/`).
- The cluster-bootstrap reference at `downstream/bootstrap/` (per-draw CIs) is reduced
  from these substrates, so any change here must be matched by a bootstrap refresh.

## Loading

```python
from huggingface_hub import hf_hub_download
import pandas as pd, json

parquet = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "downstream/xgboost.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(parquet)
print(df.shape, df.columns.tolist())

# Display + diagnostic sidecar (incl. fallback_rate)
meta_p = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "downstream/xgboost.meta.json",
    repo_type="dataset",
)
print(json.loads(open(meta_p).read()))
# -> {"display_name": "XGBoost", "type": "Statistical", ..., "fallback_rate": 0.0}
```

## Pooled substrate

The pooled per-user frame across all methods is the concatenation of the per-method
parquets (93,528 rows/method for the canonical 32-task config):

```python
import glob, pandas as pd
pooled = pd.concat(
    [pd.read_parquet(p) for p in glob.glob("downstream/*.parquet")],
    ignore_index=True,
)
```

## `fallback_rate`

Each sidecar carries `fallback_rate` — the fraction of the method's
test predictions the harness left non-finite and substituted with the `linear` baseline
before scoring. `wbm` is the only non-zero method (it embeds only participants with a
full weekly window); the rest are `0.0`. A high rate means the headline scores partly
reflect the baseline's performance on the substituted cells and should be read with
caution.

## Refreshing

The substrate is produced and uploaded from the OpenMHC code repo. It is pooled from
saved eval predictions (no model re-run); the bootstrap reference is kept on the same
predictions.

```bash
# (1) Eval — run each method through the public API, saving per-(method, task)
#     test predictions + the shared _subgroups.json.
METHOD=xgboost MHC_DATA_DIR=<data> PREDICTIONS_DIR=results/eval/final/predictions \
  python scripts/run_eval.py                      # repeat for the 8 methods

# (2) Bootstrap-draws reference for the CIs (n_boot=1000, seed=42, baseline=linear).
PYTHONPATH=src python scripts/paper_results/downstream/bootstrap_downstream_draws.py \
  --predictions_dir results/eval/final/predictions --csvs_dir results/eval/final \
  --methods linear multirocket xgboost lsm2 gru_d wbm toto chronos2 \
  --output results/paper/bootstrap_draws.parquet

# (3) Build the per-method substrate parquets (+ provenance sidecars).
python scripts/paper_results/downstream/parity/produce_per_method_per_user_pairs.py \
  --predictions-dir results/eval/final/predictions --out-dir results/leaderboard_downstream

# (4) Parity gate — the substrate must equal the predictions, and a substrate-driven
#     bootstrap must reproduce results/paper/bootstrap_draws.parquet.
python scripts/paper_results/downstream/parity/parity_substrate.py

# (5) Upload the per-method substrates (HF auth required).
for m in multirocket xgboost lsm2 gru_d wbm toto chronos2; do
  python tools/upload_leaderboard_substrate.py --dir results/leaderboard_downstream \
    --method "$m" --track downstream --name "<Display>" --type "<Type>" \
    --submitter "OpenMHC team" --subtrack static
done

# (6) Upload the bootstrap-draws reference (-> downstream/bootstrap/).
python tools/upload_leaderboard_bootstrap.py --dir results/paper --track downstream
```

Steps (2) and (6) keep the bootstrap CIs on the same canonical predictions as the point
numbers — run them together whenever the substrates change.
