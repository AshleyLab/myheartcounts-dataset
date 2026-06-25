# Downstream (Track 1) submission substrate — schema

This is the schema of the file a **submitter uploads** for the Track-1 (outcome
prediction) leaderboard: one per-method, per-user **prediction-pair** parquet
plus a small display sidecar.

```
downstream/<method>.parquet      # per-user substrate (this file)
downstream/<method>.meta.json    # display sidecar (below)
```

The parquet is produced by the evaluation run — pass both `output_dir=` and
`method_name="<method>"` to `openmhc.evaluate_prediction` and it writes
`<method>.parquet` with the `method` column set to `<method>`. Upload that file
verbatim; do not hand-author it. The maintainers concatenate every
`downstream/*.parquet` and recompute the paired skill score, fair skill score,
and average rank against the `linear` baseline during ingestion, so the columns
and dtypes below must match exactly.

> **Why pairs and not a per-user error?** Unlike imputation/forecasting (whose
> per-user MAE is a clean per-user quantity), Track-1's headline metrics — binary
> AUPRC, ordinal Spearman, regression Pearson — are cohort-level
> ranking/correlation metrics that do **not** decompose into one error per user.
> So the substrate ships the raw per-user prediction pairs and the leaderboard
> recomputes the paired metrics server-side from them. (This is a deliberate
> cross-track schema difference: Track 1 ships pairs, Tracks 2/3 ship `E`.)

> **`method_name` is required and must equal your `<method>` stem.** It defaults
> to `model.name`, and ingestion **groups rows by the `method` column** — a
> mismatch is scored under the wrong identity. This is the `method_name`
> argument to `evaluate_prediction` (which sets the parquet column), *not* the
> cosmetic `to_submission_yaml(method_name=...)` display name.

> For the separate **bootstrap reference** frame (per-draw CIs, not the submission
> file), see [`bootstrap/SCHEMA.md`](bootstrap/SCHEMA.md).

## `<method>.parquet`

One row per `(method, task, task_type, subgroup_attr, subgroup_value, user_id)`.
Single Parquet, dictionary-encoded string columns, `float32` pair values, `zstd`
compression.

| column | type | description |
|---|---|---|
| `method` | string (dict) | your method identifier; **must equal the `<method>` filename stem** — set it via `evaluate_prediction(method_name="<method>")` (defaults to `model.name`) |
| `task` | string (dict) | benchmark task name (one of the 32 `BENCHMARK_TASKS`) |
| `task_type` | string (dict) | `binary`, `ordinal`, `multiclass`, or `regression` |
| `subgroup_attr` | string (dict) | `all` (global cell), `age_group`, or `sex` |
| `subgroup_value` | string (dict) | `all` for the global cell; otherwise the subgroup level (age bucket, sex value, or `unknown`) |
| `user_id` | string (dict) | participant id from the canonical split |
| `y_true` | float32 | ground-truth label for this `(task, user)` |
| `y_pred` | float32 | the model's discrete prediction (class for binary/ordinal/multiclass; point value for regression) |
| `y_proba` | float32 | the model's continuous score — class-1 probability for binary, point prediction otherwise (the column the ranking/correlation metrics read) |

### Value semantics

- **Missing-prediction fallback is already applied.** A participant the model
  could not produce a finite prediction for has been scored against the `linear`
  baseline before this file is written (the fraction substituted is reported in
  `fallback_rate` — see the sidecar). There are no NaN pairs.
- **Structurally-absent tasks are omitted, not NaN-filled.** A task with no
  eligible test cohort simply has no rows.

### Subgroup rows are required for fairness

Every `(task, user)` cell appears three times: once with `subgroup_attr = "all"`
(the global cell used for skill and rank) and once for each demographic attribute
(`age_group`, `sex`) the user belongs to. The fairness skill score is computed
only from the `age_group` / `sex` rows, so a submission that ships only
`subgroup_attr = "all"` will score `NaN` fairness.
`evaluate_prediction(output_dir=...)` emits all three automatically.

## `<method>.meta.json`

Tiny display sidecar the leaderboard reads to render the row, produced by
`to_submission_yaml(...)`:

```jsonc
{
  "display_name": "My Method",      // shown in the leaderboard
  "type": "Deep Learning",          // category label
  "submitter": "Stanford CS",       // lab / team
  "subtrack": "static",             // static | longitudinal
  "fallback_rate": 0.0              // fraction substituted with the Linear baseline (optional; "n/a" if absent)
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-1 baseline for skill / fairness: `linear` (server-side; do not submit it).
- Must use the standard evaluation protocol — canonical dataset version, split
  file, task set, and label-validity criterion.

## Uploaded with

Submitters open a PR on the dataset with `HfApi().upload_folder(...,
create_pr=True)` (see the repo README, "Submit to the Leaderboard").
Maintainers can use `tools/upload_leaderboard_substrate.py --track downstream`
in the [code repo](https://github.com/AshleyLab/myheartcounts-dataset).
