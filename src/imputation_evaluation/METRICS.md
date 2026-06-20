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

All per-scenario scopes use the **3-level B.2 form**: L3 within-bucket
task mean → L2 mean over buckets-present within the scenario. ``n_tasks``
reports the number of buckets present in the scenario.

| scope | buckets present | n_tasks |
|---|---|---|
| `random_noise` | activity, physiology, sleep, workouts | 4 |
| `temporal_slice` | activity, physiology, sleep, workouts | 4 |
| `signal_slice` | activity, physiology, sleep, workouts | 4 |
| `sleep_gap` | activity, physiology (binary excluded by `EXCLUDE_BINARY_SCENARIOS`) | 2 |
| `workout_gap` | physiology (only `ch_5, ch_6` masked) | 1 |
| `intensity_failure` | physiology (only `ch_5, ch_6` masked) | 1 |

In single-bucket scenarios (`workout_gap`, `intensity_failure`) the
3-level form degenerates to the per-channel geomean over `ch_5, ch_6`,
so those values are numerically equivalent to a flat 2-task geomean.

### 5.2 Per-category scopes (4 rows per method per split)

Channel partition (`paper_metrics_core.py:48-53`):

| category | channels | what |
|---|---|---|
| `activity` | `ch_0..ch_4` | iPhone steps/distance/flights + Watch steps/distance |
| `physiology` | `ch_5..ch_6` | Watch heart rate + active energy |
| `sleep` | `ch_7..ch_8` | sleep asleep / inbed (binary) |
| `workouts` | `ch_9..ch_18` | 10 workout-type binary channels |

| scope | aggregation | n_tasks |
|---|---|---|
| `cat:activity` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `{ch_0..ch_4}` per-task R | 15 |
| `cat:physiology` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `{ch_5, ch_6}` per-task R | 6 |
| `cat:sleep` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:sleep` per-scenario R | 3 |
| `cat:workouts` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:workouts` per-scenario R | 3 |

All four are **structural-only** by design — `EXCLUDE_BINARY_SCENARIOS`
strips binary tasks from semantic scenarios, and continuous categories
deliberately mirror the same restriction so all four category scores
share the same scenario denominator.

**Binary `cat:sleep` / `cat:workouts`** read from the synthetic
`cat_collapsed:<cat>` channel keys in the upstream per-user dict
(`pair_aggregator.py:401-419`). Each per-user category E is the
arithmetic mean of `(1 − AUC_channel)` across that user's defined
channels in the category, after which the standard paired-ratio
machinery applies. The per-channel `cat:sleep` / `cat:workouts`
that previously took the geomean over individual binary channels
have been deleted — they overweighted workouts at 10× sleep by
construction.

**Continuous categories are not collapsed** (`activity` has 5
channels, `physiology` has 2 — they "already weight roughly fairly").
So there is no `cat_collapsed:activity` / `cat_collapsed:physiology`
synthetic channel; the bucket-source for `cat:activity` / `cat:physiology`
is the per-channel rows directly.

### 5.3 Cross-scenario scopes (2 rows per method per split)

Both use the same 3-level B.2 form as the per-scenario scopes,
with an additional **L1 log-space mean over scenarios in scope**.
``n_tasks`` reports the number of scenarios contributing.

| scope | scenarios in L1 | n_tasks |
|---|---|---|
| `semantic` | `{sleep_gap, workout_gap, intensity_failure}` | 3 |
| `overall` | all 6 scenarios | 6 |

`overall` is the headline skill / rank quoted on the leaderboard JSON
(`build_leaderboard_json.OVERALL_SKILL_SCOPE`). The legacy per-channel
`overall` (flat geomean over all 68 per-channel tasks) was deleted in
C3 of the B.2-everywhere consolidation; the new `overall` is the
3-level form universally applied. The fairness CSV's `overall` row is
a separate quantity — the cross-attribute macro of the per-attribute
disparity-ratio skill scores — and is read via
`OVERALL_FAIR_SCOPE = "overall"` from a different CSV.

### 5.3.1 The universal 3-level B.2 form

All cross-scenario scopes (`semantic`, `overall`) and all per-scenario
scopes share one operator:

```
L3 (within bucket × scenario):  bucket_log_R[m, s, b] = mean over tasks(s, b)
                                                       of log(clip(R_task))
L2 (within scenario):           scenario_log_R[m, s]  = mean over buckets-present
                                                       of bucket_log_R[m, s, b]
L1 (across scenarios):          scope_log_R[m]        = mean over scenarios-in-scope
                                                       of scenario_log_R[m, s]
S_scope[m]                       = 1 − exp(scope_log_R[m])
```

Per-scenario scopes (`random_noise`, `temporal_slice`, `signal_slice`,
`sleep_gap`, `workout_gap`, `intensity_failure`) use only L3+L2 — there
is no L1 axis because there's only one scenario. `semantic` uses L1
over the 3 semantic scenarios; `overall` uses L1 over all 6 scenarios.

Each of the four sensor categories contributes once per scenario via L2,
regardless of how many constituent channels it has:

where `C = {activity, physiology, sleep, workouts}`, `K = |C ∩ buckets present|`,
and the per-bucket geomeaned ratio is

```
R_c  =  exp(  (1/n_c) · Σ_{r ∈ tasks(c)}  log(R_task_r)  )
```

— at the per-scenario grain. The **bucket source** is fixed
(`paper_metrics_core.b2_bucket_for_channel`):
- `activity` ← continuous per-channel rows for `ch_0..ch_4`.
- `physiology` ← continuous per-channel rows for `ch_5..ch_6`.
- `sleep` ← `cat_collapsed:sleep` synthetic channel key (one row per
  scenario, value `nanmean(1 − AUC_channel)` over `ch_7..ch_8`).
- `workouts` ← `cat_collapsed:workouts` synthetic channel key (one row
  per scenario, value `nanmean(1 − AUC_channel)` over `ch_9..ch_18`).
- Per-channel binary rows (`ch_7..ch_18` with `channel_type == "binary"`)
  are **dropped** — they reach the headline only via the collapsed rows
  so per-channel binary would double-count.

Why 3-level (and not flat): a flat geomean over the underlying ~38 tasks
would weight activity 6.7× heavier than sleep (20 vs. 3 tasks) and weight
scenarios proportional to their task counts. The 3-level form gives each
bucket equal voice within each scenario and each scenario equal voice in
the headline.

Rank side mirrors the skill side: L3 per-(scenario, bucket) mean of
`task_rank`; L2 mean over buckets-present per scenario; L1 arithmetic
mean over scenarios in scope (no log/exp because rank is linear).
``n_tasks`` reports the same axis it does on the skill side — buckets
for per-scenario scopes, scenarios for cross-scenario scopes.

### 5.4 Per-task leaf scopes (`task:<s>:<c>`)

One row per `(method, scenario, channel)` cell in §5.1 plus one per
`(method, scenario, cat_collapsed:K)` synthetic-channel cell:

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

**Per-attribute skill (two-stage form, mirrors §5.3.1 — buckets only;
fairness does not have an L1 over scenarios):**

```
S^{G}_m  =  1  −  exp(  (1 / K) · Σ_{c ∈ C}  log ratio^{G}_{m, c}  )
```

where `C = {activity, physiology, sleep, workouts}`, `K = |C ∩ buckets present|`,
and each bucket-level disparity log ratio is the **per-task mean of
log(ratio_{m, r, G})** over the bucket's constituent (scenario, channel)
tasks:

```
log ratio^{G}_{m, c}  =  (1 / n_c) · Σ_{r ∈ tasks(c)}  log clip(D_{r, j}^{(G)} / D_{r, b}^{(G)},  ℓ,  u)
```

Bucket sourcing matches the skill / rank side
(`paper_metrics_core.b2_bucket_for_channel`):

- `activity` and `physiology` come from per-channel continuous rows
  (`ch_0..ch_4` and `ch_5..ch_6`).
- `sleep` and `workouts` come from the `cat_collapsed:sleep` and
  `cat_collapsed:workouts` rows (Part D — collapsed per-scenario tasks).
- Per-channel binary rows (`ch_7..ch_18`) are dropped: the sleep /
  workouts buckets reach the headline only via the collapsed rows, so
  per-channel binary would double-count.

The two-stage form prevents the same task-count imbalance the skill
side carried before — under flat per-task geomeaning the 10 binary
workout channels would weigh workouts at 30 / 68 ≈ 44% of the per-attribute
geomean, eclipsing both continuous categories combined.

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

For binary channels, the code precomputes the per-(user, channel) pooled
AUC matrix once per `(method, scenario, cell)` and reuses it across the
per-channel `E`, paired-ratio `R`, collapsed-binary, and rank reducers
(`_per_user_auc_from_cell_stats`, wired in the Phase 1 loop at
`bootstrap_skill_rank.py:399-434, 1355-1363`).

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
SE          =  nanstd_b    X^{(b)}     (ddof=1)
ci_lo, ci_hi = percentile_b X^{(b)} at (α/2, 1 − α/2)  with  α = 0.05
n_boot      =  count of finite draws in this cell
```

(`_summarize` in `bootstrap.py`, reused by Phase 2.)

CI level is 0.95 by default (set in `sweep_methods.yaml:ci_level`).
Pairing via the shared resample matrix controls sampling covariance, but it
does **not** correct the shape bias of the `max_g E − min_g E` disparity
ratio that powers the fairness skill score. Fairness rows therefore
additionally carry a deterministic `point` estimate plus a BCa
(bias-corrected & accelerated) CI alongside the percentile columns
(§S7). Skill / rank rows stay percentile-only by default — they are
near-unbiased, so BCa would not move them much — but a
`bca_skill_rank` knob exists for parity sanity-checks.

Fairness bootstrap follows the same shape: each draw computes
`(D_j, D_b)` per task, ratio per task, geomean per attribute, macro
average. SE / CI from `_summarize` over the 1000 draws
(`aggregate_fairness_skill_score.py`). The fairness CSV columns are
`[method, scope, split, n_tasks, mean, se, ci_lo, ci_hi, n_boot,
point, bca_lo, bca_hi]`; `mean / ci_lo / ci_hi` retain their legacy
percentile semantics, and `point / bca_lo / bca_hi` carry the BCa
augmentation for the 3 published fairness scopes (`overall`,
`age_group`, `sex`).

### 8.1 Recomputing against a subset of methods

The Phase 0 → Phase 1 → Phase 2 boundary is well-defined:
`bootstrap_draws.parquet` is the canonical Phase-1 output (plus its sibling
`per_user_errors.parquet`, the BCa LOO substrate emitted by
`bootstrap_imputation_draws.py --per-user-errors`), and any subset of the
methods inside them can be re-aggregated in Phase 2 alone — no Phase 0
(per-method eval) or Phase 1 (resample) rerun needed. The BCa point + LOO
jackknife are themselves per-method recomputes against `baseline_method`
(same invariant as the skill / fairness scores), so subset reruns stay
bit-identical for the in-subset rows.

Both Phase 2 aggregators take a `--method-filter` flag that pre-filters
the parquet rows before any reducer runs:

```bash
python scripts/paper_results/aggregate_imputation_paper_metrics.py \
  --draws  ${PAPER_OUT}/bootstrap_draws.parquet \
  --output-dir ${PAPER_OUT}/subset-A/ \
  --method-filter locf lsm2 lsm2_weekly_sparse linear brits \
  --baseline-method locf \
  --clip-lower 0.01 --clip-upper 100.0 \
  --lambda-fairness 0.5 --fairness-combine linear_penalty \
  --ci-level 0.95

python scripts/paper_results/aggregate_fairness_skill_score.py \
  --draws  ${PAPER_OUT}/bootstrap_draws.parquet \
  --output ${PAPER_OUT}/subset-A/fairness_skill_score_bootstrap.csv \
  --method-filter locf lsm2 lsm2_weekly_sparse linear brits \
  --baseline-method locf --clip-lower 0.01 --clip-upper 100.0 --ci-level 0.95
```

Or via the pipeline driver: set `method_filter: [locf, lsm2, …]` in the
sweep YAML and run `run_paper_pipeline.py --skip-eval --skip-phase1`.
Both aggregators are dispatched with the filter automatically.

**Two invariants the math gives you:**

| Quantity | Depends on the pool? | Why |
|---|---|---|
| Skill score (vs. baseline) | **No** | Paired ratio `E_method / E_baseline` per task is independent of every other method in the parquet |
| Fairness skill score | **No** | Disparity ratio `D_method / D_baseline` per task is also independent |
| Avg rank | **Yes** | `compute_average_rankings` ranks methods against each other per (scenario, channel, user); changing the pool changes every rank |
| Bootstrap SE / CI | follows the underlying quantity | Skill / fairness SE unchanged; rank SE depends on pool size |

So a subset rerun lets you re-cut ranks against a custom comparison
group while skill / fairness numbers stay bit-identical to the full-pool
run.

**Two operational guardrails:**

1. **The baseline must be in the filter.** Both skill and fairness pair
   each method's E against the baseline's E per task; if the baseline
   isn't in the parquet rows after filtering, the inner merge produces
   no rows and the CSV gets empty skill / fairness rows for every
   method. The pipeline driver's `_method_filter_args` helper raises a
   `ValueError` before subprocess launch if the YAML's `method_filter`
   excludes `baseline_method`; the direct CLI does not enforce this, so
   include `locf` (or whatever `--baseline-method` is) explicitly.

2. **`build_leaderboard_json.py` does NOT re-filter to its own
   `DAILY_META` / `PERSONALIZED_META` registries when reading the
   subset CSV.** It looks for every method registered in those dicts and
   raises `SystemExit` if any is missing from the CSV. If your subset
   doesn't include every method the renderer expects, either edit the
   dicts to match the subset, or invoke the renderer with a different
   working directory whose registry matches.

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

## §S7. BCa (bias-corrected & accelerated) CIs for fairness skill

**Why.** The fairness skill score reduces a per-task max-min disparity
ratio `D_{r}^{(G)} = max_g E_{r}^{(g)} − min_g E_{r}^{(g)}` clipped and
geomean-averaged across tasks. `max − min` is a skewed, downward-biased
statistic: the bootstrap mean `mean_b S^{(G),(b)}` sits below the
deterministic point estimate `Ŝ^{(G)}`, and the plain percentile CI
brackets 0 for most mid-pack methods. The shared-resample-matrix
pairing in §8 addresses sampling **covariance**, not the shape **bias**
of `max − min` — those are orthogonal concerns. BCa re-anchors the
interval at `Ŝ^{(G)}` and corrects bias + skew (second-order accurate)
without changing the reported point estimate.

**Math.** For each headline `(method, scope)` cell:

```
draws         θ*_b  =  bootstrap S^{(G),(b)} from Phase 2
point         θ̂      =  compute_fair_skill_scores(per_user → per_cell mean)
jackknife    θ_{(i)} =  leave-one-user-out recompute of the point flow

z_0           =  Φ⁻¹( frac of draws < θ̂ )                # bias correction
a             =  Σ_i d_i^3 / (6 · (Σ_i d_i^2)^{3/2}),  d = mean(θ_{(i)}) − θ_{(i)}
                                                          # acceleration
α_lo, α_hi    =  Φ( z_0 + (z_0 + z_{α/2})  / (1 − a (z_0 + z_{α/2})) )
                 Φ( z_0 + (z_0 + z_{1−α/2})/ (1 − a (z_0 + z_{1−α/2})) )
bca_lo, hi    =  percentile_b θ*_b at (100 · α_lo, 100 · α_hi)
```

`Φ` / `Φ⁻¹` come from `statistics.NormalDist` (stdlib, no scipy).

**Guards** (fall back to the plain percentile interval):

- 0 finite draws or non-finite point → percentile (or NaN, NaN if
  draws are empty).
- All draws equal → `[point, point]` (degenerate).
- Non-finite `z₀` or `a`; zero or non-finite denominator
  `1 − a (z₀ + z_q)` → percentile.
- `z₀ = a = 0` (symmetric draws, symmetric jackknife) → adjusted
  percentiles collapse to `α/2` and `1 − α/2`: the BCa interval
  equals the percentile interval exactly.

**Headline scope gating.** BCa is gated to the
headline-scope set (default ON for fairness):

| Table                                     | Headline scopes for BCa                        | Default |
|-------------------------------------------|------------------------------------------------|---------|
| `fairness_skill_score_bootstrap.csv`      | `overall`, `age_group`, `sex` (all 3 emitted)  | **ON**  |
| `skill_scores_bootstrap.csv` (opt-in)     | `overall` + sensor categories                  | off     |
| `avg_rankings_bootstrap.csv` (opt-in)     | `overall` + sensor categories                  | off     |

`point` is filled for every row (headline + non-headline); `bca_lo` /
`bca_hi` are NaN outside the headline set. Per-channel rows therefore
keep the percentile CI only, even when the opt-in flag is on.

**Skill / rank opt-in caveat.** The `--bca-skill-rank` flag on
`aggregate_imputation_paper_metrics.py` is wired through the CLI but
the LOO jackknife of the skill / rank point flow is not yet
implemented (each LOO recompute would run
`compute_per_task_paired_R` + `compute_skill_scores` +
`compute_average_rankings` once per user, ~N × per-iter at N ≈ 5K
users in the production cohort). Turning the flag on currently raises
`NotImplementedError` with a pointer back here. The flag is OFF by
default so the legacy skill / rank CSVs are unchanged.

**Backward compatibility.** The legacy columns
`[method, scope, split, n_tasks, mean, se, ci_lo, ci_hi, n_boot]`
keep their exact percentile semantics; `--no-bca` produces a CSV
byte-identical to the pre-BCa output (asserted by
`tests/imputation_evaluation/test_imputation_bca.py::test_fairness_bca_off_matches_legacy_csv`).

**Implementation pointers.**

- `src/imputation_evaluation/evaluation/bca.py` — `_jackknife_acceleration`,
  `_bca_interval`, `_augment_with_bca` (stdlib only).
- `src/imputation_evaluation/evaluation/bootstrap_skill_rank.py` —
  `compute_per_draw_errors(emit_per_user_errors=True)` and the
  `write_per_user_errors_parquet` / `read_per_user_errors_parquet`
  helpers (Phase 1 substrate).
- `scripts/paper_results/aggregate_fairness_skill_score.py` —
  `_jackknife_fair_points_from_per_user`, `_per_user_to_per_cell_E`,
  `compute_fairness_skill_scores(bca=True, per_user_df=...)`.

## See also

- [`README.md`](README.md) — public API + library usage
- [`evaluation/paper_metrics_core.py`](evaluation/paper_metrics_core.py) — pure-pandas reducers
- [`evaluation/bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py) — Phase 1 + Phase 2 kernels
- [`scripts/paper_results/run_paper_pipeline.py`](../../scripts/paper_results/run_paper_pipeline.py) — end-to-end driver
- [Forecasting Track 3 metric spec](../forecasting_evaluation/) — the parity target for the paired ratio / per-user rank reducers
