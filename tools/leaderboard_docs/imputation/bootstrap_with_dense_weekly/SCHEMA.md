# Imputation bootstrap reference — schema (17-method)

Identical structure to [`imputation/bootstrap/SCHEMA.md`](../bootstrap/SCHEMA.md);
the only difference is the method pool. This variant adds
**`lsm2_weekly`** (dense 7-day LSM-2) for 17 methods total.

The 17-method bootstrap reference is two files:

```
imputation/bootstrap_with_dense_weekly/draws.parquet      # ~116 MB zstd
imputation/bootstrap_with_dense_weekly/draws.meta.json    # provenance sidecar
```

## `draws.parquet`

One row per `(method, scenario, split, channel, channel_type,
subgroup_attr, subgroup_value, draw)`. Approximate size: **~12.1M rows**
for the 1000-draw / 17-method / 6-scenario / test-split run; `~116 MB`
compressed (`zstd`).

| column | type | description |
|---|---|---|
| `method` | string (dict) | method identifier (**17 values**: `locf`, `mean`, `mode`, `linear`, `temporal_mean`, `temporal_mode`, `personalized_mean`, `personalized_mode`, `personalized_temporal_mean`, `brits`, `dlinear`, `dlinear_weekly`, `fedformer`, `timesnet`, `lsm2`, `lsm2_weekly_sparse`, `lsm2_weekly`) |
| `scenario` | string (dict) | masking scenario (6): `random_noise`, `temporal_slice`, `signal_slice`, `sleep_gap`, `workout_gap`, `intensity_failure` |
| `split` | string (dict) | data split — `test` |
| `channel` | string (dict) | `ch_0`..`ch_18` (19 sensor channels) or `cat_collapsed:sleep` / `cat_collapsed:workouts` (collapsed-binary tasks) |
| `channel_type` | string (dict) | `continuous` (`ch_0`–`ch_6`), `binary` (`ch_7`–`ch_18`), or `binary_collapsed` (the `cat_collapsed:*` tasks) |
| `subgroup_attr` | string (dict) | `all` (global cell), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (age bucket, sex value, or `unknown`) |
| `draw` | int32 | bootstrap-draw index in `[0, n_boot)` |
| `E` | float32 | absolute error for this draw, user-macro reduced |
| `R` | float32 | paired ratio vs baseline (`locf`) for this draw, geomean over users; `1.0` exactly for the baseline-vs-self row, `NaN` if undefined |
| `rank` | float32 | cross-method rank for this draw, averaged over the resampled cohort (**range 1–17** because the comparison pool has 17 methods, vs 1–16 in the canonical bootstrap) |

### Value semantics

Same as the canonical bootstrap — see
[`imputation/bootstrap/SCHEMA.md`](../bootstrap/SCHEMA.md#value-semantics).

The only column whose values differ from the canonical variant is `rank`,
because the rank is computed across the methods in the pool and the pool
size is 17 here vs 16 there. `E` and `R` for each `(method, scenario,
split, channel, subgroup, draw)` row that exists in both files are
byte-identical.

## `draws.meta.json`

Same shape as the canonical variant. `methods` is the 17-element list
including `lsm2_weekly`. See
[`imputation/bootstrap/SCHEMA.md#drawsmeta_json`](../bootstrap/SCHEMA.md).

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-2 baseline for skill / fairness: `locf`.
- Format: single Parquet, dictionary-encoded categoricals, `float32` errors,
  `int32` draw index, `zstd` compression.
- Resamples are paired across methods — same `boot_idx` matrix per split,
  so `R` and `rank` are valid per-draw comparisons.

## When to use which

- **`imputation/bootstrap/`** (16 methods): canonical for the published
  paper. Use for cross-paper comparisons and for the leaderboard's
  rank-based scores against the historical method set.
- **`imputation/bootstrap_with_dense_weekly/`** (this dir): use when
  comparing LSM-2 dense-weekly against the other methods, or when you want
  the full set of methods that exist in the code repo as of today.
