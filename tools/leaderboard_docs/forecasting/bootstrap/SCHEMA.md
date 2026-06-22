# Forecasting bootstrap reference — schema

The Track-3 bootstrap reference is two files:

```
forecasting/bootstrap/draws.parquet      # zstd
forecasting/bootstrap/draws.meta.json    # provenance sidecar
```

## `draws.parquet`

One row per `(reduction, model, scope, metric, draw)` — a single long frame
holding the per-draw values for all three reductions.

| column | type | description |
|---|---|---|
| `reduction` | string (dict) | which reduction the row belongs to: `skill`, `rank`, or `fairness` |
| `model` | string (dict) | method identifier (10 values; see `draws.meta.json:methods`) |
| `scope` | string (dict) | the headline scope (see below) |
| `metric` | string (dict) | scored metric for `rank` rows (`mae` / `auroc` / `overall`); empty `""` for `skill` and `fairness` |
| `draw` | int32 | bootstrap-draw index in `[0, n_boot)` |
| `value` | float32 | the per-draw value of this reduction for `(model, scope[, metric])` |

### `scope` values by reduction

- **skill**: `channel_0_score`..`channel_18_score`, `sleep_score`, `workout_score`,
  `activity_score`, `physiology_score`, `overall_score`.
- **rank**: `channel_<i>`, `sleep`, `workout`, `activity`, `physiology`, `overall`
  (paired with `metric` ∈ `mae` / `auroc` / `overall`).
- **fairness**: `age_group`, `sex`, `overall`, the 4 sensor categories
  (`activity` / `physiology` / `sleep` / `workout`), and `channel_<i>`.

### `value` semantics

- **skill** — paired skill score `1 − exp(mean_task log R)` vs `seasonal_naive`,
  per draw (resampled-user cohort). `R` is the clipped per-task error ratio;
  continuous error = MAE, binary error = `max(1 − AUROC, 0.005)`.
- **rank** — cross-method average rank for the draw (lower error → rank 1),
  meaned over the resampled cohort.
- **fairness** — MAPD disparity-ratio fair skill score for the draw:
  per-task disparity `D = mean(|E_g − E_g'|)` over subgroup pairs, ratio vs the
  baseline's `D`, clipped, geomean-averaged across tasks (category-balanced),
  macro-averaged across attributes.

Resamples are **paired** across methods (one shared `boot_idx` matrix, `seed=42`),
so per-draw cross-method comparisons (skill ratios, ranks) are valid.

## `draws.meta.json`

```jsonc
{
  "n_boot": 1000,
  "seed": 42,
  "ci_level": 0.95,
  "splits": ["test"],
  "baseline": "seasonal_naive",
  "methods": ["seasonal_naive", "autoARIMA", ...],   // 10 entries
  "continuous_metrics": ["mae"],
  "binary_metrics": ["auroc"],
  "age_bins": [18, 30, 40, 50, 60],
  "reductions": ["skill", "rank", "fairness"],
  "within_user_aggregation": "micro",
  "aggregation_unit": "user",
  "n_rows": 0,
  "git_commit": "...",
  "timestamp": "..."
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-3 baseline for skill / fairness: `seasonal_naive`.
- Fairness disparity primitive: **MAPD** (mean absolute pairwise difference);
  for a 2-level attribute (`sex`) this equals the historical max-min.
- Format: single Parquet, dictionary-encoded categoricals, `float32` value,
  `int32` draw index, `zstd` compression.

## Tracks

| dir | track | status |
|---|---|---|
| `imputation/bootstrap/` | Track 2 — Imputation | live |
| `forecasting/bootstrap/` | Track 3 — Forecasting (above) | live |
| `downstream/bootstrap/` | Track 1 — Outcome Prediction | added later |
