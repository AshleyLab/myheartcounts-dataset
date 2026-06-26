# OpenMHC leaderboard substrate — schema

Each file is one method's **per-user, per-task substrate** for one track, stored as a single Parquet file at `<track>/<method>.parquet`.

These are the inputs the leaderboard recompute consumes to produce paired skill scores, cross-method ranks, and fairness skill scores. They are *reduced* per-user values — **not** raw sensor data. The three tracks differ in what each row stores: Track 1 stores raw prediction **pairs**, Track 2 stores an **error**, Track 3 stores the **raw metric value**.

## Track 1 — Predictive Tasks

One row per `(method, task, task_type, subgroup_attr, subgroup_value, user_id)`. Track-1's headline metrics — binary AUPRC, ordinal Spearman, regression Pearson — are **cohort-level** ranking/correlation statistics that do **not** decompose into one error per user. So this substrate stores the raw per-user **prediction pairs** and the leaderboard recomputes the paired metrics server-side.

| column | type | description |
|---|---|---|
| `method` | string (dict) | method identifier, e.g. `xgboost` |
| `task` | string (dict) | prediction task name (one of the 32 tasks) |
| `task_type` | string (dict) | `binary`, `ordinal`, or `regression` |
| `subgroup_attr` | string (dict) | `all` (global), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (an age bucket, or a sex value) |
| `user_id` | string (dict) | pseudonymous participant id — the bootstrap cluster unit |
| `y_true` | float32 | ground-truth label for the `(task, user)` cell |
| `y_pred` | float32 | the model's discrete prediction (class for binary/ordinal; point value for regression) |
| `y_proba` | float32 | the model's continuous score — class-1 probability for binary, point prediction otherwise (the column the ranking/correlation metrics read) |

### Why pairs, not an error

Because the metrics are cohort-level (AUPRC / Spearman / Pearson), a single per-user error like Track 2's `E_per_user` is not well-defined. The leaderboard pools every `downstream/*.parquet` and recomputes the paired skill score, fair skill score, and average rank against the `linear` baseline. Missing-prediction fallback is already applied — a participant the model could not score is substituted with the `linear` baseline before the pairs are written — so there are no NaN pairs.

## Track 2 — Imputation

One row per `(method, scenario, split, channel, channel_type, subgroup_attr, subgroup_value, user_id)`.

| column | type | description |
|---|---|---|
| `method` | string (dict) | method identifier, e.g. `locf` |
| `scenario` | string (dict) | masking scenario (6): `random_noise`, `temporal_slice`, `signal_slice`, `sleep_gap`, `workout_gap`, `intensity_failure` |
| `split` | string (dict) | data split — `test` |
| `channel` | string (dict) | `ch_0`..`ch_18` (19 sensor channels), or `cat_collapsed:sleep` / `cat_collapsed:workouts` (collapsed-binary tasks) |
| `channel_type` | string (dict) | `continuous` (`ch_0`–`ch_6`), `binary` (`ch_7`–`ch_18`), or `binary_collapsed` (the `cat_collapsed:*` tasks) |
| `subgroup_attr` | string (dict) | `all` (global), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (an age bucket, or a sex value) |
| `user_id` | string (dict) | pseudonymous participant id — the bootstrap cluster unit; required for paired skill + per-user rank |
| `E_per_user` | float32 | per-user error for the cell (see below) |

### `E_per_user` definition

- **Continuous** channels (`ch_0`–`ch_6`): per-user MAE = `Σ|y − ŷ| / N` over the user's masked positions.
- **Binary** channels (`ch_7`–`ch_18`): `1 − AUC` (per-user pooled AUC). **Unfloored** here — the `ε = 0.005` floor is applied only inside the paired-ratio reducer, not in this artifact.
- **Collapsed-binary** (`cat_collapsed:sleep` = channels 7–8; `cat_collapsed:workouts` = channels 9–18): `nanmean` over the category's channels of `1 − AUC`.

NaN values are dropped — a user with no data for a cell simply has no row.

## Track 3 — Forecasting

One row per `(model, group, metric, channel_idx, channel_name, user_id)`. Unlike Track 2 (which stores the error `E_per_user`), the forecasting substrate stores the **raw** per-user `metric_value`, so a single file serves all three reducers — skill and fairness convert it to an error on load, rank uses it directly.

| column | type | description |
|---|---|---|
| `model` | string (dict) | method identifier, e.g. `seasonal_naive` |
| `group` | string (dict) | `continuous` (channels 0–6, scored on `mae`) or `binary` (channels 7–18, scored on `auroc`) |
| `metric` | string (dict) | scored metric: `mae` (continuous) or `auroc` (binary) |
| `channel_idx` | int16 | sensor channel index (0–18) |
| `channel_name` | string (dict) | human-readable channel name |
| `user_id` | string (dict) | pseudonymous participant id — the bootstrap cluster unit |
| `metric_value` | float64 | RAW per-user metric, micro-pooled over the user's forecast windows (`Σcell / Σcount`) |
| `n_values` | int64 | finite horizon-cell count behind `metric_value` |

### From `metric_value` to error

The skill/fairness reducers convert on load: continuous error = `metric_value` (MAE, lower=better); binary error = `max(1 − auroc, 0.005)`. Rank uses `metric_value` directly. Stored at **float64** (vs Track 2's float32) for byte-exact reproduction of the published aggregates.

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Baselines for skill / fairness: `linear` (Track 1), `locf` (Track 2), `seasonal_naive` (Track 3). The baseline is scored server-side; do not submit it.
- Track 3 within-user aggregation: micro (pool finite horizon cells); unit: user.
- Fairness disparity primitive: mean absolute pairwise difference (MAPD) — for a 2-level attribute (`sex`) this equals the historical max-min.
- Format: one Parquet per method, dictionary-encoded categoricals; `float32` pairs (Track 1) / `float32` errors (Track 2) / `float64` `metric_value` (Track 3).
- Per-track bootstrap references live under `<track>/bootstrap/` — see that subdir's `SCHEMA.md`.
- Per-track details (full schema, submission steps): see `<track>/SCHEMA.md`.

## Tracks

| dir | track | status |
|---|---|---|
| `downstream/` | Track 1 — Predictive Tasks (above) | live |
| `imputation/` | Track 2 — Imputation (above) | live |
| `forecasting/` | Track 3 — Forecasting (above) | live |
