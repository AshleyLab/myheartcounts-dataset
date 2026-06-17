# Track 2 — Imputation Metric Specification

Canonical formulas and scope set for the OpenMHC imputation leaderboard.
All implementations live in
[`paper_metrics_core.py`](evaluation/paper_metrics_core.py) and
[`bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py); the run
scripts in [`scripts/paper_results/`](../../scripts/paper_results/) are
thin orchestrators.

## TL;DR

The leaderboard publishes three families of numbers per method per split:

| File | Quantity | Reducer | Code |
|---|---|---|---|
| `skill_scores_bootstrap.csv` | **Skill score** (regret vs. baseline) | paired per-user geomean | `compute_skill_scores` |
| `avg_rankings_bootstrap.csv` | **Average rank** (lower is better) | per-user rank, two-stage | `compute_average_rankings` |
| `fairness_skill_score_bootstrap.csv` | **Fairness skill score** (disparity ratio) | per-attribute geomean | `compute_fair_skill_scores` |

Every cell carries `mean / se / ci_lo / ci_hi / n_boot` from a 1000-draw
cluster bootstrap over users.

## Notation

| Symbol | Meaning |
|---|---|
| `m` | method (e.g. `lsm2`, `linear`, `locf`) |
| `b` | baseline method (default: `locf`, see `BASELINE_CONTINUOUS`) |
| `s` | masking scenario (one of 6) |
| `c` | channel (one of 19, or one of 2 collapsed-binary categories) |
| `r = (s, c)` | task |
| `u` | user (cluster unit for the bootstrap) |
| `g` | subgroup level (e.g. `30-39`, `female`) within attribute `G` ∈ {age_group, sex} |
| `E_{m,r,u}` | per-task per-user error (defined below) |
| `R_{m,r}` | per-task paired ratio against the baseline |
| `S_{m, scope}` | skill score for one scope |
| `T_{m,r}` | per-task rank of method `m` on task `r` (averaged over users) |

## 1. Per-task per-user error E

Per-task per-user errors are produced upstream by
[`pair_aggregator.aggregate_pairs(..., return_per_user=True)`](evaluation/pair_aggregator.py).

**Continuous channels (`ch_0..ch_6`, 7 channels)**

```
E_{m,r,u}  =  MAE  =  ( Σ_t |y_t − ŷ_t| ) / N_{r,u}
```

over the masked positions `(r, u)` (subset of `target_mask == 1`). MAE
matches Track 3 forecasting — the per-task ratio is invariant under MAE
vs RMSE vs nMAE, so the skill score doesn't depend on this choice
(`paper_metrics_core.py:99-107`).

**Per-channel binary channels (`ch_7..ch_18`, 12 channels)**

```
E_{m,r,u}  =  1 − AUC_{m,r,u}     (per-user pooled AUC, then floored)
E_{m,r,u}  ←  max(E_{m,r,u}, BINARY_ERROR_FLOOR)
```

The floor `ε = 0.005` (`bootstrap_skill_rank.py:80`) prevents a
perfect-AUC user from dividing the paired ratio by zero and being
dropped — matches forecasting Track 3.

**Collapsed-binary "channels" (Part D — `cat_collapsed:sleep`, `cat_collapsed:workouts`)**

To avoid the 10-workout-channels-vs-2-sleep-channels weight imbalance under
the per-task geomean, two synthetic tasks are added per scenario:

```
E_{m, s, cat_collapsed:K, u}  =  nanmean_{c ∈ K} (1 − AUC_{m, s, c, u})
```

with the same floor applied after the nanmean
(`pair_aggregator.py:401-419`,
`bootstrap_skill_rank.py:_per_method_cell_collapsed_errors`).

## 2. Per-task paired ratio R

```
r_{m,r,u}  =  clip( E_{m,r,u} / E_{b,r,u},  CLIP_LOWER,  CLIP_UPPER )
R_{m,r}    =  exp(  mean_u  log r_{m,r,u}  )                  (geomean over users)
```

Constants: `CLIP_LOWER = 0.01`, `CLIP_UPPER = 100.0`
(`paper_metrics_core.py:25-26`; same values aliased as
`SKILL_CLIP_LOWER` / `SKILL_CLIP_UPPER` in
`bootstrap_skill_rank.py:77-78`).

Users are dropped per task if `E_{b,r,u} ≤ 0` or non-finite; tasks
require `SKILL_MIN_PAIRS = 1` surviving user. The baseline-vs-self row
(`m = b`) is excluded — `R_{b,r} ≡ 1` by construction (see
`compute_per_task_paired_R` at
`bootstrap_skill_rank.py:888-985`).

## 3. Skill score S

**Per task (leaf scope `task:<s>:<c>`)** — degenerate single-task case:

```
S^{task}_{m, s, c}  =  1 − clip(R_{m, s, c}, CLIP_LOWER, CLIP_UPPER)
```

**Per aggregated scope** — geomean of clipped ratios over the scope's
tasks:

```
S_{m, scope}  =  1 − exp(  mean_{r ∈ scope}  log clip(R_{m,r}, CLIP_LOWER, CLIP_UPPER)  )
```

Higher is better; `S = 0` ⇔ method matches baseline; `S > 0` ⇔ method
beats baseline. Code: `compute_skill_scores(mode="paired")` at
`paper_metrics_core.py:143-341`.

The reducer additionally accepts `mode="pooled"` (legacy
`E_method / E_baseline` on user-macro E) but this is no longer the
leaderboard estimand and is kept only for the deprecated
`S − λ·D` fairness loop.

## 4. Average rank T

Two-stage form, mirrors forecasting Track 3
(`_average_rankings_per_user` at `paper_metrics_core.py:420-474`):

**Stage 1 — per-user rank, then user mean (`task_rank`)**

For each task `r = (s, c)`:

1. Build the `users × methods` matrix of E values.
2. Rank methods within each user (`method="average"`, ascending — lowest
   E → rank 1, ties averaged).
3. `task_rank_{m,r} = nanmean_u rank_{m,r,u}`

**Stage 2 — mean over tasks in scope**

```
T_{m, scope}  =  mean_{r ∈ scope}  task_rank_{m, r}
```

Per-user ranking is scale-free, so channels with different absolute MAE
magnitudes don't bias the cross-task mean.

**Per task (leaf scope `task:<s>:<c>`)** — degenerate single-task case
(emitted directly from Stage 1):

```
T^{task}_{m, s, c}  =  task_rank_{m, s, c}
```

## 5. Scope catalog

The same scope set is emitted by `compute_skill_scores` and
`aggregate_task_ranks_to_scopes`. Each row in the output CSVs is keyed
by `(method, scope, split)`.

### 5.1 Per-scenario scopes (6 rows per method per split)

| scope | task set | task count |
|---|---|---|
| `random_noise` | `ch_0..ch_18` | 19 |
| `temporal_slice` | `ch_0..ch_18` | 19 |
| `signal_slice` | `ch_0..ch_18` | 19 |
| `sleep_gap` | `ch_0..ch_6` (binary excluded — see §6) | 7 |
| `workout_gap` | `ch_5, ch_6` (only masked channels — see §6) | 2 |
| `intensity_failure` | `ch_5, ch_6` (only masked channels — see §6) | 2 |

### 5.2 Per-category scopes (4 rows per method per split)

Channel partition (`paper_metrics_core.py:48-53`):

| category | channels | what |
|---|---|---|
| `activity` | `ch_0..ch_4` | iPhone steps/distance/flights + Watch steps/distance |
| `physiology` | `ch_5..ch_6` | Watch heart rate + active energy |
| `sleep` | `ch_7..ch_8` | sleep asleep / inbed (binary) |
| `workouts` | `ch_9..ch_18` | 10 workout-type binary channels |

| scope | task set | task count |
|---|---|---|
| `cat:activity` | `{random_noise, temporal_slice, signal_slice}` × `{ch_0..ch_4}` | 15 |
| `cat:physiology` | `{random_noise, temporal_slice, signal_slice}` × `{ch_5, ch_6}` | 6 |
| `cat:sleep` | `{random_noise, temporal_slice, signal_slice}` × `{ch_7, ch_8}` | 6 |
| `cat:workouts` | `{random_noise, temporal_slice, signal_slice}` × `{ch_9..ch_18}` | 30 |

All four are **structural-only** by design — `EXCLUDE_BINARY_SCENARIOS`
strips binary tasks from semantic scenarios, and continuous categories
deliberately mirror the same restriction so all four category scores
share the same scenario denominator
(`paper_metrics_core.py:42-45`, `:273-281`).

### 5.3 Collapsed-binary scopes (2 rows per method per split, Part D)

Geometric mean over the two binary categories with each category
collapsed to a single task per scenario (see §1 for the per-task E):

| scope | task set | task count |
|---|---|---|
| `cat_collapsed:sleep` | `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:sleep` | 3 |
| `cat_collapsed:workouts` | `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:workouts` | 3 |

Same formula as §3, computed over the collapsed-binary R values
(`paper_metrics_core.py:300-308`).

**Motivation** — without collapsing, the per-channel geomean weights
`workouts` 10× `sleep` (10 vs. 2 binary channels), and in any pooled
scope the binary side is dominated by workout channels. Collapsing makes
each binary category count once per scenario, so sleep and workouts
weigh equally and don't swamp the 7 continuous channels in `overall`.
Continuous categories are deliberately **not** collapsed (`activity` has
5 channels, `physiology` has 2 — they "already weight roughly fairly"
per `paper_metrics_core.py:60-64`), so there is no `cat_collapsed:activity`
or `cat_collapsed:physiology`. Only sleep and workouts have collapsed
twins (`BINARY_CATEGORIES_ORDERED`, `:65-68`).

**Leaderboard consumption** — the leaderboard JSON's `sleep` and
`workout` columns read `cat_collapsed:sleep` and `cat_collapsed:workouts`,
not the per-channel `cat:sleep` / `cat:workouts`
(`build_leaderboard_json.SUBGROUP_FIELD`). The per-channel `cat:sleep` /
`cat:workouts` scopes remain in the CSVs as secondary references.

### 5.4 Cross-scenario scopes (3 rows per method per split)

| scope | task set | task count |
|---|---|---|
| `semantic` | `{sleep_gap, workout_gap, intensity_failure}` × applicable channels | 7 + 2 + 2 = 11 |
| `overall` | all 6 scenarios × all per-channel tasks (binary excluded from semantic) | 19·3 + 7 + 2 + 2 = **68** |
| `overall_binary_collapsed` | category-balanced two-stage geomean over **4 buckets** | **4** |

`overall_binary_collapsed` is the headline skill / rank quoted on the
leaderboard JSON (`build_leaderboard_json.OVERALL_SKILL_SCOPE`).
`overall` is kept as a secondary per-channel reference and is the scope
the fairness CSV's macro row uses (it has no collapsed variant — only
one disparity per attribute).

### 5.4.1 The B.2 two-stage form for `overall_binary_collapsed`

Each of the four sensor categories contributes **once** to the headline,
regardless of how many constituent (channel × scenario) tasks live inside it:

```
S_overall_binary_collapsed  =  1  −  exp(  (1/K) · Σ_{c ∈ C}  log(R_c)  )
```

where `C = {activity, physiology, sleep, workouts}`, `K = |C ∩ buckets present|`,
and the per-bucket geomeaned ratio is

```
R_c  =  exp(  (1/n_c) · Σ_{r ∈ tasks(c)}  log(R_task_r)  )
```

— exactly the value that `cat:<c>` (continuous) or `cat_collapsed:<c>`
(binary) already report.

The **bucket source** is fixed:
- `activity` and `physiology` come from `cat:<c>` (continuous per-channel
  scopes, structural scenarios only).
- `sleep` and `workouts` come from `cat_collapsed:<c>` (Part D — collapsed
  per-scenario tasks). The per-channel `cat:sleep` / `cat:workouts` scopes
  are **deliberately not consumed** here — that's the whole point of the
  binary collapse.

Why two-stage: a flat geomean over the underlying ~38 tasks would weight
activity (20 tasks) ~6.7× heavier than sleep (3 collapsed tasks). The
two-stage form removes that imbalance and gives each modality equal voice
in the headline.

Rank side mirrors the skill side: the per-bucket value is the bucket's
arithmetic mean of `task_rank` (i.e. the `avg_rank` of the corresponding
`cat:*` / `cat_collapsed:*` scope), and the headline rank is the arithmetic
mean across the 4 buckets — `n_tasks` again counts buckets present.

### 5.5 Per-task leaf scopes (`task:<s>:<c>`)

One row per `(method, scenario, channel)` cell in §5.1 plus one per
`(method, scenario, cat_collapsed:K)` cell in §5.3:

```
68  per-channel task scopes
 +  6  collapsed-binary task scopes
= 74 leaf scopes per method per split
```

For these leaves `n_tasks = 1`,
`skill_score = 1 − clip(R)` (single-task case of the geomean formula),
`avg_rank = task_rank` (single-task case of the Stage-2 mean).

## 6. Per-scenario channel inventory (correctness contract)

The pair writer only writes rows for actually-masked positions, so each
scenario's task set is determined by its mask generator's
`mask_channels` argument
([`masking/*.py`](masking/)):

| Scenario | Mask channels | Evaluated channels |
|---|---|---|
| `random_noise` | random patches per channel | `ch_0..ch_18` |
| `temporal_slice` | contiguous blocks, all channels | `ch_0..ch_18` |
| `signal_slice` | whole channels per day | `ch_0..ch_18` |
| `sleep_gap` | all except `ch_7, ch_8` during sleep | `ch_0..ch_6`*, `ch_9..ch_18`*\*\* |
| `workout_gap` | `ch_5, ch_6` during workouts | `ch_5, ch_6` |
| `intensity_failure` | `ch_5, ch_6` when HR > threshold | `ch_5, ch_6` |

\* Binary `ch_9..ch_18` rows are produced but dropped by
`EXCLUDE_BINARY_SCENARIOS` before any aggregated scope — semantic
scenarios are continuous-only in the leaderboard.

This restriction is inherited automatically by `task:*` scopes; no
per-scenario allowlist exists in the kernel.

## 7. Fairness skill score (disparity ratio)

Implemented in `_per_attribute_skill_keyed` at
`paper_metrics_core.py:525-621` and `compute_fair_skill_scores` at
`:624-708`; bootstrapped by
[`scripts/paper_results/aggregate_fairness_skill_score.py`](../../scripts/paper_results/aggregate_fairness_skill_score.py).

**Per-task per-attribute disparity:**

```
D_{m, r, G}  =  max_g  E_{m, r, g}  −  min_g  E_{m, r, g}
```

where `g` ranges over the levels of attribute `G ∈ {age_group, sex}`
(see `DEFAULT_FAIRNESS_ATTRS` at `paper_metrics_core.py:35`).

**Per-task disparity ratio vs. baseline:**

```
ratio_{m, r, G}  =  clip( D_{m, r, G} / D_{b, r, G},  CLIP_LOWER,  CLIP_UPPER )
```

Tasks are dropped from the `G`-aggregation if (i) fewer than two common
subgroups exist for both `m` and `b` on task `r`, (ii) `D_{b, r, G} ≤ 0`
or NaN, or (iii) `D_{m, r, G}` is NaN.

**Per-attribute skill:**

```
S^{G}_m  =  1  −  exp(  mean_r  log ratio_{m, r, G}  )
```

**Macro-average over attributes:**

```
S^{fair}_m  =  (1 / |A|) · Σ_{G ∈ A}  S^{G}_m
```

Methods missing data for any attribute drop out of the macro to keep the
average honest (`paper_metrics_core.py:688-690`).

**Emitted scopes** (3 rows per method per split):

| scope | value |
|---|---|
| `age_group` | `S^{age_group}_m` |
| `sex` | `S^{sex}_m` |
| `overall` | `S^{fair}_m` |

The fairness CSV is **not** broken down by scenario / category / channel
— that's tracked in [the plan
file](/home/users/schuetzn/.claude/plans/have-a-look-at-squishy-kernighan.md)
as an out-of-scope follow-up.

## 8. Bootstrap (cluster-on-user)

**Phase 1 — `bootstrap_imputation_draws.py`**

For each split, draw `n_boot = 1000` resample matrices `M ∈ N_users × n_boot`,
each column a multiset bootstrap over the canonical user set.
The matrix is **shared across all scenarios and subgroups within the
split** so cross-scenario covariance and within-draw subgroup pairing
are preserved (`bootstrap_skill_rank.py:_seed_for_split:384-396`,
seeded by `SHA-256("seed|split")` so the draw is reproducible across
runs).

For each draw `(b)`:
- recompute per-task per-user paired ratios on the resampled cohort
- aggregate to `R^{(b)}_{m, r}` (geomean over users, §2)
- compute per-task per-user ranks → `task_rank^{(b)}_{m, r}` (Stage 1, §4)
- AUC for binary channels uses the per-user pooled Mann-Whitney U with
  multiplicity from `M` (`_bootstrap_auc_from_arrays` at
  `bootstrap_skill_rank.py:309-376`)

Output: `bootstrap_draws.parquet` with one row per
`(method, scenario, split, channel, subgroup_attr, subgroup_value, draw)`
carrying `E`, `R`, `rank`.

**Phase 2 — `aggregate_imputation_paper_metrics.py`**

Calls `compute_skill_scores(mode="paired")` and
`aggregate_task_ranks_to_scopes` once per draw, stacks the per-draw scope
tables, then for each `(method, scope, split)` cell summarizes across
draws:

```
mean        =  nanmean_b   X^{(b)}
SE          =  nanstd_b    X^{(b)}     (ddof=0)
ci_lo, ci_hi = percentile_b X^{(b)} at (α/2, 1 − α/2)  with  α = 0.05
n_boot      =  count of finite draws in this cell
```

(`_summarize` in `bootstrap.py`, reused by Phase 2.)

CI level is 0.95 by default (set in `sweep_methods.yaml:ci_level`); the
percentile method is bias-uncorrected — we rely on the
shared-resample-matrix structure to control pairing rather than
attempting an analytic acceleration term.

Fairness bootstrap follows the same shape: each draw computes
`(D_j, D_b)` per task, ratio per task, geomean per attribute, macro
average. SE / CI from `_summarize` over the 1000 draws
(`aggregate_fairness_skill_score.py`).

## 9. Constants (verified canonical values)

| Constant | Value | Location |
|---|---|---|
| `CLIP_LOWER` / `SKILL_CLIP_LOWER` | `0.01` | `paper_metrics_core.py:25`, `bootstrap_skill_rank.py:77` |
| `CLIP_UPPER` / `SKILL_CLIP_UPPER` | `100.0` | `paper_metrics_core.py:26`, `bootstrap_skill_rank.py:78` |
| `BASELINE_CONTINUOUS` / `BASELINE_BINARY` | `"locf"` | `paper_metrics_core.py:28-29` |
| `BINARY_ERROR_FLOOR` | `0.005` | `bootstrap_skill_rank.py:80` |
| `SKILL_MIN_PAIRS` | `1` | `bootstrap_skill_rank.py:79` |
| `EXCLUDE_BINARY_SCENARIOS` | `{sleep_gap, workout_gap, intensity_failure}` | `paper_metrics_core.py:42` |
| `SEMANTIC_SCENARIOS` | `{sleep_gap, workout_gap, intensity_failure}` | `paper_metrics_core.py:45` |
| `DEFAULT_FAIRNESS_ATTRS` | `("age_group", "sex")` | `paper_metrics_core.py:35` |
| Default age bins | `[18, 30, 40, 50, 60]` | `sweep_methods.yaml`, `run_paper_pipeline.py:186` |
| `n_boot` | `1000` | `sweep_methods.yaml:37` |
| `seed` | `42` | `sweep_methods.yaml:38` |
| CI level | `0.95` | `sweep_methods.yaml:51` |
| Splits scored | `["test"]` | `sweep_methods.yaml:39` |

## 10. Worked example — `task:random_noise:ch_5` for one method

Inputs (illustrative, 3 users):

| user | E_{m, random_noise, ch_5, u} (MAE) | E_{b, random_noise, ch_5, u} (MAE) |
|---|---|---|
| u1 | 4.0 | 8.0 |
| u2 | 2.0 | 4.0 |
| u3 | 1.0 | 10.0 |

Per-user ratios (clip range `[0.01, 100]`, all in-bounds):

```
r_u1 = 4.0 / 8.0   = 0.50
r_u2 = 2.0 / 4.0   = 0.50
r_u3 = 1.0 / 10.0  = 0.10
```

Per-task R (geomean over users):

```
R = exp( (log 0.50 + log 0.50 + log 0.10) / 3 )
  = exp( (−0.693 + −0.693 + −2.303) / 3 )
  = exp( −1.230 )
  = 0.292
```

Per-task skill (task:* leaf scope):

```
S^{task}_{m, random_noise, ch_5}  =  1 − clip(0.292, 0.01, 100)
                                   =  1 − 0.292
                                   =  0.708
```

Per-scenario skill — geomean over `random_noise`'s 19 channels'
clipped Rs, then `1 − exp(mean(log R))`. The task:* leaf is the
single-task case where the mean collapses to one term.

Bootstrap: repeat the per-user join with each draw's resampled cohort,
get 1000 values of `S^{task,(b)}`, report
`mean / se / ci_lo / ci_hi`.

## See also

- [`README.md`](README.md) — public API + library usage
- [`evaluation/paper_metrics_core.py`](evaluation/paper_metrics_core.py) — pure-pandas reducers
- [`evaluation/bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py) — Phase 1 + Phase 2 kernels
- [`scripts/paper_results/run_paper_pipeline.py`](../../scripts/paper_results/run_paper_pipeline.py) — end-to-end driver
- [Forecasting Track 3 metric spec](../forecasting_evaluation/) — the parity target for the paired ratio / per-user rank reducers
