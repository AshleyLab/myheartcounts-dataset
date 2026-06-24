# Imputation bootstrap reference — schema

The Track-2 bootstrap reference is two files:

```
imputation/bootstrap/draws.parquet      # ~107 MB zstd
imputation/bootstrap/draws.meta.json    # provenance sidecar
```

## `draws.parquet`

One row per `(method, scenario, split, channel, channel_type,
subgroup_attr, subgroup_value, draw)`. Approximate size: **11,374,208
rows** for the canonical 1000-draw / 16-method / 6-scenario / test-split
run; `~107 MB` compressed (`zstd`).

| column | type | description |
|---|---|---|
| `method` | string (dict) | method identifier (16 values: `locf`, `mean`, `mode`, `linear`, `temporal_mean`, `temporal_mode`, `personalized_mean`, `personalized_mode`, `personalized_temporal_mean`, `brits`, `dlinear`, `dlinear_weekly`, `fedformer`, `timesnet`, `lsm2`, `lsm2_weekly_sparse`) |
| `scenario` | string (dict) | masking scenario (6): `random_noise`, `temporal_slice`, `signal_slice`, `sleep_gap`, `workout_gap`, `intensity_failure` |
| `split` | string (dict) | data split — `test` |
| `channel` | string (dict) | `ch_0`..`ch_18` (19 sensor channels) or `cat_collapsed:sleep` / `cat_collapsed:workouts` (collapsed-binary tasks) |
| `channel_type` | string (dict) | `continuous` (`ch_0`–`ch_6`), `binary` (`ch_7`–`ch_18`), or `binary_collapsed` (the `cat_collapsed:*` tasks) |
| `subgroup_attr` | string (dict) | `all` (global cell), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (age bucket, sex value, or `unknown`) |
| `draw` | int32 | bootstrap-draw index in `[0, n_boot)` |
| `E` | float32 | absolute error for this draw, user-macro reduced |
| `R` | float32 | paired ratio vs baseline (`locf`) for this draw, geomean over users; `1.0` exactly for the baseline-vs-self row, `NaN` if undefined |
| `rank` | float32 | cross-method rank for this draw, averaged over the resampled cohort |

### Value semantics

- **`E`** — per-draw user-macro error. Per channel: continuous = per-user MAE, then `nanmean` over resampled users; binary = `1 − AUC` per user, then `nanmean` over users; `binary_collapsed` = per-user `nanmean(1 − AUC)` over the category's channels, then `nanmean` over users.
- **`R`** — per-draw paired ratio. Per (user, task) the ratio is `clip(E_method / E_baseline, 0.01, 100)`; per draw, `R = exp(nanmean(log(ratio)))` over resampled users. Binary `E` is `ε = 0.005` floored before forming the ratio so perfect-AUC users contribute finite values.
- **`rank`** — per-draw cross-method rank. Per (user, task) the rank is `rankdata(E_methods, method="average", ascending=True)` (lower E → rank 1); per draw, `rank = nanmean` over resampled users.

NaN appears where a task is structurally absent (e.g. binary tasks in the
`EXCLUDE_BINARY_SCENARIOS` set: `sleep_gap`, `workout_gap`,
`intensity_failure`).

## `draws.meta.json`

Side-car metadata, written next to `draws.parquet`. Schema:

```jsonc
{
  "n_boot": 1000,
  "seed": 42,
  "splits": ["test"],
  "scenarios": ["intensity_failure", "random_noise", "signal_slice",
                "sleep_gap", "temporal_slice", "workout_gap"],
  "methods": ["mean", "mode", "locf", ...],          // 16 entries
  "method_dirs": {"locf": "/path/to/pairs", ...},    // source pair dirs (cluster-local)
  "include_auc": true,
  "include_fairness": true,
  "age_bins": [18, 30, 40, 50, 60],
  "exclude_unknown": false,
  "elapsed_seconds": 3531.6,
  "n_rows": 11374208,
  "git_commit": "06e4cbda...",                       // code-repo commit at generation
  "timestamp": "2026-06-19T03:24:21.860842Z",
  "argv": [...]                                       // exact invocation
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-2 baseline for skill / fairness: `locf`.
- Format: single Parquet, dictionary-encoded categoricals, `float32` errors, `int32` draw index, `zstd` compression.
- Resamples are paired across methods — same `boot_idx` matrix per split, so `R` and `rank` are valid per-draw comparisons.

## Sibling

`imputation/bootstrap_with_dense_weekly/` holds the 17-method variant
(adds `lsm2_weekly`). Same schema as this file; `rank` ranges 1–17
instead of 1–16 because the comparison pool grew, and `methods` in
`draws.meta.json` has 17 entries. Skill / fairness values per method
are byte-identical between the two variants (pairwise vs `locf`).

## Tracks

| dir | track | status |
|---|---|---|
| `imputation/` | Track 2 — Imputation per-method substrate (`SCHEMA.md` there) | live |
| `imputation/bootstrap/` | Track 2 — Imputation bootstrap (above, 16-method) | live |
| `imputation/bootstrap_with_dense_weekly/` | Track 2 — Imputation bootstrap (17-method sibling) | live |
| `forecasting/bootstrap/` | Track 3 — Forecasting | added later |
| `downstream/bootstrap/` | Track 1 — Outcome Prediction | added later |
