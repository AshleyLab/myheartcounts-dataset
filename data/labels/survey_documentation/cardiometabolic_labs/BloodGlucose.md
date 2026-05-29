# BloodGlucose

**Benchmark column**: `field_BloodGlucose`
**Raw identifier**: `heartAgeDataBloodGlucose`
**Obj-C constant**: `kHeartAgeTestBloodGlucose`
**Role**: context
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 68
- Used in Framingham calculation: No direct use in the Framingham 10-year risk or heart age coefficient calculation
- UI question: `APHHeartAgeTaskViewController.m` lines 277-287
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Fasting Blood Glucose (optional)

## Answer options
| Value | Label |
|-------|-------|
| Variable | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: HealthKit-backed numeric field using `HKQuantityTypeIdentifierBloodGlucose`. Style is integer or decimal depending on locale. Optional field.

## Observed values

**Total observations**: 6,147 — **type-enforced**: 6,147 (**unique**: 228) — raw Python types seen: `float` (6,147).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.06 |
| q25 | 4.72 |
| median | 5.11 |
| mean | 5.22 |
| q75 | 5.55 |
| max | 54.95 |
| std | 2.07 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `5.00` | 326 |
| `5.27` | 257 |
| `4.72` | 233 |
| `5.55` | 223 |
| `4.94` | 204 |
| `4.88` | 203 |
| `5.05` | 203 |
| `5.44` | 198 |
| `4.44` | 195 |
| `5.22` | 190 |
| `5.11` | 185 |
| `5.16` | 185 |
| `4.83` | 184 |
| `5.33` | 182 |
| `5.49` | 181 |
| `4.77` | 156 |
| `5.38` | 155 |
| `4.66` | 145 |
| `4.61` | 118 |
| `4.55` | 112 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508 UK localization), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Blood glucose is collected for informational and research purposes but is **NOT directly used in the Framingham 10-year risk or heart age calculation** (no references to this constant in the Framingham coefficient calculation method).
- Marked as optional (line 283).
- Captured on the "Cholesterol & Glucose" form step, with helper text indicating optional fields should be entered as 0 if unknown (line 260).
- The HealthKit integration at line 278–280 suggests this may be populated from HealthKit if available, and the localization supports both mg/dL (US) and mmol/L (UK) units.
- Diabetes (a categorical yes/no) is the primary diabetes-related risk factor used in Framingham; glucose level is secondary/contextual.
