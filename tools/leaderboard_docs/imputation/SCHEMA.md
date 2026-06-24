# Imputation substrate — schema

The Track-2 (imputation) leaderboard substrate is a set of per-method parquet
files plus per-method JSON sidecars at the top level of the `imputation/` dir:

```
imputation/<method>.parquet         # ~3 MB per method, zstd
imputation/<method>.meta.json       # display + diagnostic sidecar
```

These are the canonical inputs that downstream leaderboard renderers and
re-aggregators consume. The bootstrap reference (`imputation/bootstrap/` and
its sibling `imputation/bootstrap_with_dense_weekly/`) is reduced from these
substrates via `scripts/paper_results/imputation/bootstrap_imputation_draws.py`.

## `<method>.parquet`

One row per `(method, scenario, split, channel, channel_type,
subgroup_attr, subgroup_value, user_id)`. Roughly **148,510 rows per method**
for the canonical config (test split, 17 methods, 6 scenarios, 19 sensor
channels + 2 collapsed-binary tasks, 3 subgroup attributes).

| column | type | description |
|---|---|---|
| `method` | string | method identifier (matches `<method>.parquet` filename without the `pypots_` prefix used internally) |
| `scenario` | string | masking scenario: `random_noise`, `temporal_slice`, `signal_slice`, `sleep_gap`, `workout_gap`, `intensity_failure` |
| `split` | string | data split — `test` (val is for development only and is not part of the leaderboard substrate) |
| `channel` | string | `ch_0`–`ch_18` (19 sensor channels) or `cat_collapsed:sleep` / `cat_collapsed:workouts` (collapsed-binary tasks) |
| `channel_type` | string | `continuous` (`ch_0`–`ch_6`), `binary` (`ch_7`–`ch_18`), or `binary_collapsed` (the `cat_collapsed:*` tasks) |
| `subgroup_attr` | string | `all` (global cell), `age_group`, or `sex` |
| `subgroup_value` | string | `all` for the global cell; otherwise the subgroup level (age bucket `18-29`/`30-39`/`40-49`/`50-59`/`60+`/`unknown`, or `male`/`female`/`unknown` for sex) |
| `user_id` | string | per-user identifier from the OpenMHC canonical split |
| `E_per_user` | float32 | per-user error — `continuous` = per-user MAE; `binary` = `1 − AUC` (un-floored); `binary_collapsed` = `nanmean` over the category's channels of `1 − AUC[user, ch]`. NaN rows are dropped before the parquet is written. |

Values match the bootstrap's `_per_user_errors_for_cell` semantics, so
re-aggregating across methods (rank, skill, fairness) produces identical
numbers to the bootstrap reference's E column collapsed over draws.

## `<method>.meta.json`

Display + diagnostic sidecar consumed by the leaderboard renderer.

```jsonc
{
  "display_name": "BRITS",         // shown in the leaderboard table
  "type": "Neural",                // "Statistical" / "Neural" / submitter-defined
  "submitter": "OpenMHC team",     // attribution
  "subtrack": "single-day",        // "single-day" / "long-context" / "other"
  "fallback_rate": 0.0             // worst-case overall_fallback_rate (see below)
}
```

### `fallback_rate`

Scalar in `[0, 1]`. The worst-case `overall_fallback_rate` across all
`(scenario, split)` cells in the method's run (mirrors
`openmhc._results.ImputationResults.overall_fallback_rate`).

- **`0.0`** — every cell the model was asked to predict produced a finite
  value; the harness never substituted a fallback.
- **`>0`** — the model returned a non-finite value at that fraction of
  target cells and the harness substituted a channel-aware global baseline
  before scoring. Treat headline scores cautiously when `fallback_rate >
  ~5%` since the metric is inflated with baseline performance on the
  substituted positions.

The field is added to the sidecar by `tools/upload_leaderboard_substrate.py`
in one of two ways:

1. Explicit: `--fallback-rate FLOAT`
2. Auto: `--results-json PATH` — the tool reads the method's
   `results.json` and extracts `max(scenarios[*][*].overall_fallback_rate)`.

Existing sidecar fields are preserved on update (the tool fetches the
current sidecar from HF and merges only the provided fields).

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (test only).
- Method names match the keys in `configs/paper/sweep_methods.yaml`.
- Format: single Parquet per method, dictionary-encoded categoricals, `float32`
  errors, `zstd` compression.

## Tracks

| dir | track | status |
|---|---|---|
| `imputation/` (this doc) | Track 2 — Imputation per-method substrate | live |
| `imputation/bootstrap/` | Track 2 — Imputation cluster-bootstrap reference (`SCHEMA.md` there) | live |
| `imputation/bootstrap_with_dense_weekly/` | Sibling 17-method variant (includes `lsm2_weekly` dense) | live |
| `forecasting/` | Track 3 — Forecasting per-method substrate | live |
| `downstream/` | Track 1 — Outcome Prediction | added later |
