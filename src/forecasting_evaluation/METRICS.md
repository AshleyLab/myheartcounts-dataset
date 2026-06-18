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

Per-user, scale-free ranks (lower error → better rank), aggregated **mean-of-ranks,
users-first** — mirroring the imputation track (`_average_rankings_per_user` +
`aggregate_task_ranks_to_scopes`) and the skill score's user-first collapse
(`grouped_metric_rank_summary.py`):

- **per channel `(scope, metric)`** (`_compute_mean_ranks`): rank models within each
  user, then average over users → one task rank per channel.
- **per sensor category & `overall`** (`_compute_category_balanced_ranks`): average the
  per-channel task ranks within each category (activity/physiology/sleep/workout), then
  average the 4 categories equally for the `overall` headline (so the 10 workout channels
  can't dominate). Because users are collapsed first (per-channel task rank = mean over
  users) and then ranks are averaged — never rank-of-pooled-mean — mixed-scale channels
  within a category (e.g. steps≈1000s vs flights≈1s) don't bias the result. `overall` is
  emitted under a synthetic `metric = overall` (a cross-scope headline spans both metric
  families).

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

### Point estimate + BCa interval (fairness)

The fairness disparity `D = max_g E − min_g E` is a **range** statistic: resampling
noise inflates it, so its bootstrap **mean sits below the point estimate** and the
percentile CI is **biased low** (it brackets 0 for much of the mid-pack). For the
headline scopes the fairness table therefore reports the **point estimate** plus a
**BCa (bias-corrected & accelerated) 95% CI** — extra columns `point, bca_lo, bca_hi`
alongside the unchanged percentile columns. The reported value stays the point; BCa
only re-anchors the interval near it and corrects bias + skew (second-order accurate).
Per `(model, scope)`, from point `θ̂`, draws `θ*_b` (B), and an **exact** leave-one-
user-out jackknife `θ₍ᵢ₎`:

```
z0  = Φ⁻¹( #{θ*_b < θ̂} / B )                          bias correction (prop clipped to [0.5/B, 1−0.5/B])
d   = mean_i(θ₍ᵢ₎) − θ₍ᵢ₎ ;  a = Σ d³ / (6·(Σ d²)^{3/2})   acceleration (nan-aware; a=0 if Σd²=0)
α_q = Φ( z0 + (z0 + z_q) / (1 − a(z0 + z_q)) ) ,  z_q = Φ⁻¹(q)   for q = α/2, 1−α/2
bca = [ quantile(θ*, α_{α/2}), quantile(θ*, α_{1−α/2}) ]
```

Φ / Φ⁻¹ come from `statistics.NormalDist` (stdlib — no scipy). Guards: all draws equal
→ `[θ̂, θ̂]`; non-finite `z0`/`a` or a zero denominator `1 − a(z0+z_q)` → fall back to
the percentile interval. When `z0 = a = 0` the formula reduces exactly to the percentile
interval. Headline scopes = `overall`, the 4 categories, and the 2 attributes
(`age_group`/`sex`); per-channel scopes keep the percentile CI only.

Skill and rank are per-task **ratios/means** (mean ≈ point, near-unbiased), so they keep
the percentile CI by default; BCa for them is **opt-in** (`bootstrap.bca_skill_rank`),
mainly to confirm the machinery leaves a well-behaved metric ≈ unchanged. See
`tests/test_forecasting_bca.py`.
