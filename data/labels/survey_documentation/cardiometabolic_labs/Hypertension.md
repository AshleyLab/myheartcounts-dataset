# Hypertension

**Benchmark column**: `Hypertension`
**Raw identifier**: `heartAgeDataHypertension`
**Obj-C constant**: `kHeartAgeTestDataHypertension`
**Role**: target
**Type**: binary

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 78
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 196, 197, 227, 230, 233, 236, 290, 411, 414–417
- UI question: `APHHeartAgeTaskViewController.m` lines 341-347
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Are you being treated for Hypertension (High Blood Pressure)?

## Answer options
| Value | Label |
|-------|-------|
| 0 (NO) | No (not treated) |
| 1 (YES) | Yes (treated/on medication) |

**Input format**: Boolean, mapped to 0 (No) or 1 (Yes).

## Observed values

**Total observations**: 10,209 — **type-enforced**: 10,209 (**unique**: 2) — raw Python types seen: `str` (10,209).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 7,798 | 76.4% |
| `True` | 2,411 | 23.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Hypertension specifically refers to **medicated/treated** high blood pressure, not just elevated systolic BP.
- Systolic blood pressure is used differently in Framingham coefficients depending on this flag: treated systolic (line 196) vs. untreated systolic (line 197), with separate coefficient terms (lines 226-237).
- Also used in lifetime risk categorization (line 411): thresholds for systolic BP differ when hypertension treatment = 1.
- Set to 0 (no) in optimal risk factor scenario (line 290).
- The question emphasizes "being treated for" rather than simply having high blood pressure, reflecting the clinical significance of treatment status in cardiovascular risk models.
