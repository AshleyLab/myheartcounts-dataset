# SystolicBloodPressure

**Benchmark column**: `SystolicBloodPressure`
**Raw identifier**: `heartAgeDataSystolicBloodPressure`
**Obj-C constant**: `kHeartAgeTestDataSystolicBloodPressure`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 66
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 196, 197, 227, 230, 233, 236, 288, 410
- UI question: `APHHeartAgeTaskViewController.m` lines 302-312
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Systolic Blood Pressure

## Answer options
| Value | Label |
|-------|-------|
| 90–200 | Integer (mmHg) |

**Input format**: Numeric (integer), minimum 90, maximum 200 mmHg. Unit: mmHg.

## Observed values

**Total observations**: 9,867 — **type-enforced**: 9,867 (**unique**: 101) — raw Python types seen: `float` (9,867).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 70.00 |
| q25 | 112 |
| median | 120 |
| mean | 120.9 |
| q75 | 128 |
| max | 200 |
| std | 13.71 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `120` | 1,610 |
| `110` | 819 |
| `130` | 599 |
| `118` | 415 |
| `125` | 343 |
| `128` | 312 |
| `115` | 279 |
| `100` | 274 |
| `122` | 270 |
| `140` | 267 |
| `124` | 244 |
| `116` | 240 |
| `117` | 236 |
| `112` | 225 |
| `90.00` | 213 |
| `135` | 207 |
| `114` | 170 |
| `121` | 151 |
| `126` | 148 |
| `127` | 147 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update), `34c0781` (MHC-178 identifier fix)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Systolic BP is a primary Framingham input. It is used differently depending on hypertension treatment status.
- Two separate calculations: treated systolic BP (when `kHeartAgeTestDataHypertension` = 1, line 196) and untreated systolic BP (when hypertension = 0, line 197).
- Used as log-transformed value in Framingham coefficients (lines 227, 230, 233, 236).
- Also used in lifetime risk categorization (line 410): thresholds at 120, 140, 160 mmHg.
- Optimal systolic BP for 10-year risk is 110 mmHg (defined at line 103 in `.m` file).
- Appears on the "Blood pressure" form step, marked as required.
- Form step description: "Blood pressure (typically shown as systolic over diastolic)" (line 299).
