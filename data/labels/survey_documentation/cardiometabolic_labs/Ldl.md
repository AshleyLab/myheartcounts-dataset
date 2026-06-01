# Ldl

**Benchmark column**: `Ldl`
**Raw identifier**: `heartAgeDataLdl`
**Obj-C constant**: `kHeartAgeTestDataLDL`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 65
- Used in Framingham calculation: Not directly used in coefficient calculation; no `kHeartAgeTestDataLDL` reference in Framingham method
- UI question: `APHHeartAgeTaskViewController.m` lines 265-275
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> LDL Cholesterol (optional)

## Answer options
| Value | Label |
|-------|-------|
| 0–1000 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 0, maximum 1000. Unit is localized. Marked as OPTIONAL.

## Observed values

**Total observations**: 6,845 — **type-enforced**: 6,845 (**unique**: 226) — raw Python types seen: `float` (6,845).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.50 |
| q25 | 4.44 |
| median | 5.61 |
| mean | 5.76 |
| q75 | 6.99 |
| max | 14.93 |
| std | 2.00 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `5.55` | 178 |
| `4.44` | 140 |
| `5.00` | 113 |
| `6.66` | 112 |
| `6.11` | 104 |
| `7.21` | 96 |
| `5.49` | 91 |
| `5.72` | 88 |
| `3.89` | 84 |
| `5.27` | 84 |
| `6.38` | 83 |
| `4.72` | 80 |
| `5.88` | 80 |
| `5.22` | 78 |
| `5.38` | 78 |
| `5.61` | 78 |
| `4.16` | 77 |
| `4.83` | 75 |
| `4.94` | 75 |
| `5.11` | 74 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2HC-508)

## Notes
- LDL ("bad cholesterol") is captured for informational and research purposes, but is NOT used in the Framingham 10-year risk or heart age calculation.
- Marked as optional with helper text: "The items below are optional. If you do not know the values you need to enter 0." (line 260).
- Does not appear in the Framingham coefficients lookup or calculation.
- The app notes in the "Learn More" section that "the risk score and heart age can be affected in people taking cholesterol medications" and that results do not apply to people with LDL > 190 (lines 118–120).
