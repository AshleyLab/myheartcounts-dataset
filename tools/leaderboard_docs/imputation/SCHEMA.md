# Imputation submission substrate — schema

This is the schema of the file a **submitter uploads** for the Track-2
(imputation) leaderboard: one per-method, per-user error parquet plus a small
display sidecar.

```
imputation/<method>.parquet      # per-user substrate (this file)
imputation/<method>.meta.json    # display sidecar (below)
```

The parquet is produced by the evaluation run — pass both `output_dir=` and
`method_name="<method>"` to `openmhc.evaluate_imputation` and it writes
`per_user_errors.parquet` with the `method` column set to `<method>`. Upload
that file verbatim (renamed to `<method>.parquet`); do not hand-author it. The
maintainers concatenate every `imputation/*.parquet` and recompute paired skill,
fair skill, and average rank against the `locf` baseline during ingestion, so
the columns and dtypes below must match exactly.

> **`method_name` is required and must equal your `<method>` stem.** It defaults
> to `"custom"`, and ingestion **groups rows by the `method` column** — so a
> submission left at the default collides with every other default submission
> and is scored under the wrong identity. This is the `method_name` argument to
> `evaluate_imputation` (which sets the parquet column), *not* the cosmetic
> `to_submission_yaml(method_name=...)` display name.

> For the separate **bootstrap reference** frame (per-draw CIs, not the
> submission file), see [`bootstrap/SCHEMA.md`](bootstrap/SCHEMA.md).

## `<method>.parquet`

One row per `(method, scenario, split, channel, channel_type, subgroup_attr,
subgroup_value, user_id)`. Single Parquet, dictionary-encoded string columns,
`float32` error, `zstd` compression.

| column | type | description |
|---|---|---|
| `method` | string (dict) | your method identifier; **must equal the `<method>` filename stem** — set it via `evaluate_imputation(method_name="<method>")` (defaults to `"custom"`) |
| `scenario` | string (dict) | masking scenario (6): `random_noise`, `temporal_slice`, `signal_slice`, `sleep_gap`, `workout_gap`, `intensity_failure` |
| `split` | string (dict) | data split — `test` |
| `channel` | string (dict) | `ch_0`..`ch_18` (19 sensor channels), or `cat_collapsed:sleep` / `cat_collapsed:workouts` (collapsed-binary tasks) |
| `channel_type` | string (dict) | `continuous` (`ch_0`–`ch_6`), `binary` (`ch_7`–`ch_18`), or `binary_collapsed` (the `cat_collapsed:*` tasks) |
| `subgroup_attr` | string (dict) | `all` (global cell), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (age bucket, sex value, or `unknown`) |
| `user_id` | string (dict) | participant id from the canonical split |
| `E_per_user` | float32 | per-user error for this `(task, subgroup, user)` cell (see below) |

### Value semantics

- **`E_per_user`** — the per-user error, *before* any cross-user reduction.
  Per channel: continuous = per-user MAE; binary = `1 − AUC` for that user;
  `binary_collapsed` = the user's `nanmean(1 − AUC)` over the category's
  channels. Lower is better.
- **Structurally-absent tasks are omitted, not NaN-filled.** The binary and
  `binary_collapsed` tasks do not exist in the no-binary scenarios (`sleep_gap`,
  `workout_gap`, `intensity_failure`), so those rows are simply absent — the
  grid is not a full cartesian product. The shipped LOCF baseline
  (`src/openmhc/data/baselines/imputation_locf_per_user_errors.parquet`,
  148,510 rows, 0 NaN) is the canonical shape your submission should mirror.

### Subgroup rows are required for fairness

Every `(task, user)` cell appears three times: once with
`subgroup_attr = "all"` (the global cell used for skill and rank) and once for
each demographic attribute (`age_group`, `sex`) the user belongs to. The
fairness skill score is computed only from the `age_group` / `sex` rows, so a
submission that ships only `subgroup_attr = "all"` will score `NaN` fairness.
`evaluate_imputation(output_dir=...)` emits all three automatically.

## `<method>.meta.json`

Tiny display sidecar the leaderboard reads to render the row. Produced by
`to_submission_yaml(...)`:

```jsonc
{
  "display_name": "My Method",     // shown in the leaderboard
  "type": "Deep Learning",         // category label
  "submitter": "Stanford CS",      // lab / team
  "subtrack": "single-day"         // single-day | long-context
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-2 baseline for skill / fairness: `locf` (server-side; do not submit it).
- Must use the standard evaluation protocol — canonical dataset version, split
  file, masking configuration, and label-validity criterion.

## Uploaded with

Submitters open a PR on the dataset with `HfApi().upload_folder(...,
create_pr=True)` (see the repo README, "Submit to the Leaderboard").
Maintainers can use `tools/upload_leaderboard_substrate.py` in the
[code repo](https://github.com/AshleyLab/myheartcounts-dataset).
