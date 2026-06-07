# Diabetes

**Benchmark column**: `Diabetes`
**Raw identifier**: `heartAgeDataDiabetes`
**Obj-C constant**: `kHeartAgeTestDataDiabetes`
**Role**: target
**Type**: binary

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 70
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 245, 294
- UI question: `APHHeartAgeTaskViewController.m` lines 333-339
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Do you have Diabetes?

## Answer options
| Value | Label |
|-------|-------|
| 0 (NO) | No |
| 1 (YES) | Yes |

**Input format**: Boolean, mapped to 0 (No) or 1 (Yes).

## Observed values

**Total observations**: 10,209 — **type-enforced**: 10,209 (**unique**: 2) — raw Python types seen: `str` (10,209).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 9,726 | 95.3% |
| `True` | 483 | 4.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Diabetes is a significant risk factor in the Framingham model. When diabetes = 1, it contributes a positive coefficient to the 10-year risk (line 245).
- Used in optimal risk factor calculation: set to 0 (no) for the optimal scenario (line 294).
- Appears as a standalone question step, marked as required.
- The question step uses `ORKBooleanAnswerFormat` (line 334).
