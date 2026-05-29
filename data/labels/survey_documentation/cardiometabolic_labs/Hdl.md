# Hdl

**Benchmark column**: `Hdl`
**Raw identifier**: `heartAgeDataHdl`
**Obj-C constant**: `kHeartAgeTestDataHDL`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 64
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 195, 221, 224, 286
- UI question: `APHHeartAgeTaskViewController.m` lines 246-256
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> HDL Cholesterol

## Answer options
| Value | Label |
|-------|-------|
| 10–140 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 10, maximum 140. Unit is localized via `HKUnit.localizedCholesterolUnit`.

## Observed values

**Total observations**: 9,187 — **type-enforced**: 9,187 (**unique**: 81) — raw Python types seen: `float` (9,187).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.50 |
| q25 | 2.22 |
| median | 2.77 |
| mean | 2.86 |
| q75 | 3.50 |
| max | 5.00 |
| std | 0.99 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.11` | 462 |
| `2.77` | 434 |
| `3.33` | 380 |
| `4.44` | 345 |
| `2.22` | 308 |
| `2.50` | 297 |
| `3.05` | 223 |
| `2.33` | 218 |
| `3.61` | 205 |
| `2.66` | 197 |
| `3.89` | 193 |
| `2.61` | 192 |
| `2.55` | 191 |
| `2.28` | 187 |
| `2.11` | 185 |
| `2.89` | 185 |
| `2.44` | 183 |
| `5.00` | 178 |
| `2.39` | 176 |
| `1.94` | 174 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508 UK localization), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2024, MHC-508)

## Notes
- HDL ("good cholesterol") is inversely associated with cardiovascular risk; higher HDL is protective.
- Appears in Framingham coefficients as log-transformed value (lines 195, 221, 224).
- Optimal HDL for 10-year risk calculation is 50 (defined in lookup at line 102 in `.m` file).
- Input appears on the "Cholesterol & Glucose" form step, marked as required (not optional).
