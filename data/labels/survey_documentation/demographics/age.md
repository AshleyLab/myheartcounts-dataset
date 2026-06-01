# age

**Benchmark column**: `age`
**Raw identifier**: `heartAgeDataAge`
**Obj-C constant**: `kHeartAgeTestDataAge`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 62
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 175, 193, 209
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

**Total observations**: 57,939 — **type-enforced**: 57,939 (**unique**: 77) — raw Python types seen: `int` (57,939).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 17.00 |
| q25 | 24.00 |
| median | 32.00 |
| mean | 35.35 |
| q75 | 43.00 |
| max | 93.00 |
| std | 14.38 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `18.00` | 5,200 |
| `30.00` | 3,193 |
| `25.00` | 1,932 |
| `26.00` | 1,872 |
| `24.00` | 1,853 |
| `29.00` | 1,811 |
| `23.00` | 1,757 |
| `27.00` | 1,744 |
| `28.00` | 1,718 |
| `31.00` | 1,662 |
| `32.00` | 1,617 |
| `22.00` | 1,581 |
| `33.00` | 1,551 |
| `35.00` | 1,519 |
| `19.00` | 1,499 |
| `20.00` | 1,468 |
| `21.00` | 1,432 |
| `34.00` | 1,427 |
| `36.00` | 1,243 |
| `37.00` | 1,217 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76` (MHX-640 Added NSLocalizedString), `0869e98` (Squashed commit)
- View controller commits: `dbdd5a0` (MHC-508), `06a6f76`, `a2c3b1e` (MHC-86)
- Recent material change: `dbdd5a0` (2024, MHC-508 UK Heart Risk Task unit localization)

## Notes
- This is the self-reported age entered by the user in the initial demographics form. It is used as a primary input to the Framingham risk calculation.
- Can be pre-populated from HealthKit birth date when "Are you submitting your own heart risk data?" is answered YES (line 684 in view controller).
- The actual age appears twice: once in the demographic step and modified/corrected in the summary results before final submission (lines 500-507).
