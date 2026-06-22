# Forecasting submission substrate — schema

This is the schema of the file a **submitter uploads** for the Track-3
(forecasting) leaderboard: one per-method, per-user metric parquet plus a small
display sidecar.

```
forecasting/<method>.parquet      # per-user substrate (this file)
forecasting/<method>.meta.json    # display sidecar (below)
```

The parquet is produced by the evaluation run — pass both `output_dir=` and
`method_name="<method>"` to `openmhc.evaluate_forecasting` and it writes
`per_user_errors.parquet` with the `model` column set to `<method>`. Upload that
file verbatim (renamed to `<method>.parquet`); do not hand-author it. The
maintainers concatenate every `forecasting/*.parquet` and recompute paired skill,
fair skill, and average rank against the `seasonal_naive` baseline during
ingestion, so the columns and dtypes below must match exactly.

> **`method_name` is required and must equal your `<method>` stem.** It defaults
> to `"custom"`, and ingestion **groups rows by the `model` column** — so a
> submission left at the default collides with every other default submission and
> is scored under the wrong identity. This is the `method_name` argument to
> `evaluate_forecasting` (which sets the parquet column), *not* the cosmetic
> `to_submission_yaml(method_name=...)` display name. It must differ from the
> baseline name `seasonal_naive`.

> For the separate **bootstrap reference** frame (per-draw CIs, not the
> submission file), see [`bootstrap/SCHEMA.md`](bootstrap/SCHEMA.md).

## `<method>.parquet`

One row per `(model, group, metric, channel_idx, channel_name, user_id)`. Single
Parquet, dictionary-encoded string columns, **`float64`** metric value, `zstd`
compression. Unlike Track 2 (which stores the error `E_per_user`), the forecasting
substrate stores the **raw** per-user metric value, so one file serves all three
reducers — skill and fairness convert it to an error on load, rank uses it
directly.

| column | type | description |
|---|---|---|
| `model` | string (dict) | your method identifier; **must equal the `<method>` filename stem** — set it via `evaluate_forecasting(method_name="<method>")` (defaults to `"custom"`) |
| `group` | string (dict) | `continuous` (`ch 0`–`6`, scored on `mae`) or `binary` (`ch 7`–`18`, scored on `auroc`) |
| `metric` | string (dict) | scored metric: `mae` (continuous) or `auroc` (binary) |
| `channel_idx` | int16 | sensor channel index (0–18) |
| `channel_name` | string (dict) | human-readable channel name |
| `user_id` | string (dict) | participant id from the canonical split |
| `metric_value` | float64 | RAW per-user metric, micro-pooled over the user's forecast windows (`Σcell / Σcount`) |
| `n_values` | int64 | finite horizon-cell count behind `metric_value` |

### Value semantics

- **`metric_value`** — the raw per-user metric, *before* any cross-user reduction
  and *before* any error conversion. Continuous channels: per-user MAE. Binary
  channels: per-user AUROC. Stored at **float64** for byte-exact reproduction of
  the published aggregates.
- **On load the maintainers convert to an error**: continuous `error =
  metric_value` (MAE, lower is better); binary `error = max(1 − auroc, 0.005)`
  (the `0.005` floor keeps perfect-AUROC users finite). Rank uses `metric_value`
  directly. Skill is the paired geomean of clipped per-user error ratios vs the
  baseline.
- **Structurally-absent cells are omitted, not NaN-filled** — a `(channel,
  metric, user)` with no finite horizon cells simply has no row; the grid is not a
  full cartesian product. The shipped Seasonal-Naive baseline
  (`src/openmhc/data/baselines/forecasting_seasonal_naive_per_user_errors.parquet`)
  is the canonical shape your submission should mirror.

### No subgroup rows — fairness is joined server-side

Unlike Track 2, the forecasting substrate is keyed by `user_id` only and carries
**no `subgroup_attr` / `subgroup_value` columns**. The fairness skill score is
computed maintainer-side by joining demographics (`age_group`, `sex`) onto
`user_id` from the private label tables, so submitters do **not** ship subgroup
rows — `evaluate_forecasting` emits exactly the columns above.

## `<method>.meta.json`

Tiny display sidecar the leaderboard reads to render the row:

```jsonc
{
  "display_name": "My Forecaster",   // shown in the leaderboard
  "type": "Deep Learning",           // category label
  "submitter": "Stanford CS",        // lab / team
  "subtrack": "other"                // Track 3 has no single-day/long-context split
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-3 baseline for skill / fairness: `seasonal_naive` (server-side; do not
  submit it).
- Scored set: `mae` (continuous channels 0–6) + `auroc` (binary channels 7–18);
  ratio clip `[0.01, 100]`; within-user aggregation `micro`; unit `user`.
- Fairness disparity primitive: mean absolute pairwise difference (MAPD).
- Must use the standard evaluation protocol — canonical dataset version, split
  file, sample-index, and horizon (24 h).

## Uploaded with

Submitters open a PR on the dataset with `HfApi().upload_folder(...,
create_pr=True)` (see the repo README, "Submit to the Leaderboard").
Maintainers can use `tools/upload_leaderboard_substrate.py` in the
[code repo](https://github.com/AshleyLab/myheartcounts-dataset).
