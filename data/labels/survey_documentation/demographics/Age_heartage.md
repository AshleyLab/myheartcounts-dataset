# Age_heartage

> **Not in released benchmark.** Reason: redundancy — duplicate of target `age` (one canonical age retained). See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `Age_heartage` / `field_Age_heartage`
**Raw identifier**: `heartAgeDataAge`
**Obj-C constant**: `kHeartAgeTestDataAge`
**Role**: context
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 62
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 175, 193, 209 (same as `age` target variable)
- UI question: `APHHeartAgeTaskViewController.m` lines 153-162
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> What is your age?

## Answer options
| Value | Label |
|-------|-------|
| 18–150 | Integer (years) |

**Input format**: Numeric (integer), minimum 18, maximum 150 years

## Observed values

**Total observations**: 3,766 — **type-enforced**: 3,766 (**unique**: 74) — raw Python types seen: `float` (3,766).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 6.00 |
| q25 | 34.00 |
| median | 46.00 |
| mean | 47.48 |
| q75 | 60.00 |
| max | 89.00 |
| std | 15.68 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `34.00` | 105 |
| `42.00` | 103 |
| `30.00` | 90 |
| `43.00` | 89 |
| `35.00` | 85 |
| `45.00` | 84 |
| `28.00` | 83 |
| `38.00` | 83 |
| `50.00` | 83 |
| `33.00` | 82 |
| `29.00` | 80 |
| `39.00` | 80 |
| `40.00` | 79 |
| `41.00` | 79 |
| `53.00` | 78 |
| `52.00` | 77 |
| `32.00` | 76 |
| `44.00` | 76 |
| `49.00` | 76 |
| `31.00` | 75 |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `06a6f76`, `a2c3b1e` (MHC-86)
- Recent material change: `dbdd5a0` (2024, MHC-508)

## Notes
- This is the **same variable as the `age` target variable** above. Both refer to `heartAgeDataAge`. The distinction between `age` (target) and `Age_heartage` (context) is purely a naming convention in the benchmark/data export schema.
- Age is both a target input to the Framingham calculation and contextual information for understanding the risk assessment (the Framingham model applies primarily to ages 40–79 for 10-year risk and 20–59 for lifetime risk).
- Self-reported age from the demographic form, can be pre-populated from HealthKit birth date when user selects "Are you submitting your own heart risk data?" = YES (line 684).
- Modified and stored again in the summary results before final submission to ensure consistency (lines 500–507).
