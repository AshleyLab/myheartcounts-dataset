# Forecasting (Track 3) leaderboard metrics

How the per-channel offline metrics roll up into the published skill score, average
rank, and fairness-adjusted skill score. The scheme mirrors the imputation track
(`src/imputation_evaluation/METRICS.md`) but, because forecasting has a single task
(no masking scenarios), collapses imputation's 3-level aggregation to **two levels**:

```
per-channel task  →  sensor-category scope  →  category-balanced overall
```

The defining property: each of the 4 sensor categories is weighted **once** in the
overall, so the 10 workout channels do not dominate the headline.

## 1. Channels and scopes

The 19 channels partition into 4 semantic scopes — the single source of truth is
`CATEGORY_SCOPES` in `metrics/metric_spec.py`:

| Scope        | Channels | Kind       | Headline metric (default) |
|--------------|----------|------------|---------------------------|
| `activity`   | 0–4      | continuous | MAE                       |
| `physiology` | 5–6      | continuous | MAE                       |
| `sleep`      | 7–8      | binary     | AUROC                     |
| `workout`    | 9–18     | binary     | AUROC                     |

Metrics are configurable (`continuous_metrics` / `binary_metrics`); the canonical
run uses **MAE** (continuous) and **AUROC** (binary). Higher-is-better metrics are
converted to error `e = 1 − value` (with a small floor, `BINARY_ERROR_FLOOR`).

## 2. Per-task skill

For a task `r = (channel, metric)`, model `m`, baseline `b` (default
`seasonal_naive`), over the users common to both:

```
R_r = clip( geomean_users( E_{m} / E_{b} ), 0.01, 100 )      skill_r = 1 − R_r
```

## 3. Category-balanced aggregation (skill)

Two-stage geometric mean over **log-ratios** (`_aggregate_overall_category_balanced_score`
in `metrics/skill_score_summary.py`):

```
scope_log_R[c]  = mean over (channel, metric) tasks in scope c of log(R_r)   (Stage 1)
overall_skill   = 1 − exp( mean over the 4 scopes c of scope_log_R[c] )      (Stage 2)
```

Per-scope skill is `1 − exp(scope_log_R[c])`; per-channel skill is `1 − exp` over
that channel's metric tasks. Only the raw per-channel rows feed the overall (never a
derived scope row), so doubling a scope's channel count cannot change it.

## 4. Average rank

Per-user, scale-free ranks (lower error → better rank), then averaged
(`grouped_metric_rank_summary.py`):

- **per scope `(scope, metric)`**: rank models within each user, average over users.
- **`overall`** (`_compute_overall_category_balanced_ranks`): per user, rank models
  for each `(channel, metric)`, average within each scope, then average the 4 scopes
  equally; finally average over users. Emitted under a synthetic `metric = overall`
  (a cross-scope headline rank spans both metric families).

## 5. Fairness-adjusted skill score

Disparity-ratio fairness (`metrics/fair_skill_score.py`), category-balanced and
macro-averaged across sensitive attributes `A = {age_group, sex}`. For task `r`,
attribute `G`, subgroups `g` (the `unknown` bucket is a real subgroup):

```
D_r = max_g E_r^{(g)} − min_g E_r^{(g)}          (model and baseline, common subgroups)
ρ_r = clip( D_{r,m} / D_{r,b}, 0.01, 100 )
S^{(G)} = 1 − exp( mean_c [ mean_{r∈c} log ρ_r ] )      (category-balanced, like §3)
S_fair  = (1/|A|) · Σ_{G∈A} S^{(G)}
```

Tasks with <2 common subgroups, a perfectly-fair baseline (`D_b ≤ 0`), or any NaN
gap are dropped (prevents a single-subgroup model scoring bogus perfect fairness).
Per-scope and per-channel fairness use the inner `1 − exp(mean log ρ)` and are
**macro-averaged across attributes**, dropping any model/key missing an attribute.

## 6. Scopes and outputs

| Scope label | Meaning |
|-------------|---------|
| `channel_0` … `channel_18` | per-channel |
| `activity` / `physiology` / `sleep` / `workout` | per sensor category |
| `overall` | category-balanced headline (each category weighted once) |
| `age_group` / `sex` | per-attribute fairness (category-balanced) |

Written to `output_root` (see README Part 3); new rows ride the existing CSVs:

- `forecasting_skill_score_{long,model_summary,wide}.csv` — `model_summary` carries
  `channel_<i>_score` (all 19) + `overall_score`.
- `forecasting_grouped_metric_rank_{long,wide}.csv` — per-channel + `overall` rows.
- `forecasting_fairness_skill_score.csv` — per-channel, per-category, per-attribute,
  and `overall` rows.
- matching `*_bootstrap.csv` for each.

## 7. Bootstrap

Paired participant-level (user) bootstrap: one shared user-resample matrix applied to
every model + baseline jointly, the point-flow aggregators re-run per draw, reduced to
`mean / se / 95% CI` (`bootstrap_skill_rank.py`, `bootstrap_fair_skill_score.py`). The
identity draw reproduces the point estimate — the primary correctness gate (see
`tests/test_forecasting_bootstrap_skill_rank.py`,
`tests/test_forecasting_fair_skill_score_bootstrap.py`).
