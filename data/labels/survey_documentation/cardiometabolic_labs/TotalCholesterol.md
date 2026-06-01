# TotalCholesterol

**Benchmark column**: `TotalCholesterol`
**Raw identifier**: `heartAgeDataTotalCholesterol`
**Obj-C constant**: `kHeartAgeTestDataTotalCholesterol`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 63
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 194, 215, 218, 284, 409
- UI question: `APHHeartAgeTaskViewController.m` lines 234-244
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Total Cholesterol

## Answer options
| Value | Label |
|-------|-------|
| 80–400 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 80, maximum 400. Unit is localized via `HKUnit.localizedCholesterolUnit`.

## Observed values

**Total observations**: 9,937 — **type-enforced**: 9,937 (**unique**: 261) — raw Python types seen: `float` (9,937).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 2.00 |
| q25 | 7.77 |
| median | 9.49 |
| mean | 9.58 |
| q75 | 11.10 |
| max | 19.98 |
| std | 2.33 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `7.21` | 1,047 |
| `8.32` | 328 |
| `9.99` | 318 |
| `11.10` | 314 |
| `7.77` | 234 |
| `9.44` | 194 |
| `10.54` | 190 |
| `8.88` | 181 |
| `4.44` | 172 |
| `9.71` | 140 |
| `8.05` | 117 |
| `11.65` | 116 |
| `10.27` | 111 |
| `7.49` | 109 |
| `9.16` | 105 |
| `12.21` | 95 |
| `10.82` | 87 |
| `10.71` | 82 |
| `10.21` | 77 |
| `9.77` | 76 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508 UK localization), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2024, MHC-508)

## Notes
- Total cholesterol is a primary input to the Framingham 10-year risk calculation.
- Used as log-transformed value in coefficient calculations (lines 194, 215, 218).
- Also used in lifetime risk factor categorization (line 409): thresholds at 180, 200, 240 mg/dL.
- Optimal total cholesterol for 10-year risk is 170 (defined at line 101 in `.m` file).
- Appears on the "Cholesterol & Glucose" form step, marked as required.
