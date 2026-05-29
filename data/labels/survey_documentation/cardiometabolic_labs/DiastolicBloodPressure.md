# DiastolicBloodPressure

> **Not in released benchmark.** Reason: redundancy — DBP not a benchmark target (only `SystolicBloodPressure` is). See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `DiastolicBloodPressure` / `field_DiastolicBloodPressure`
**Raw identifier**: `heartAgeDataDiastolicBloodPressure`
**Obj-C constant**: `kHeartAgeTestDataDiastolicBloodPressure`
**Role**: context
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 67
- Used in Framingham calculation: No direct use in Framingham coefficients
- UI question: `APHHeartAgeTaskViewController.m` lines 314-326
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Diastolic Blood Pressure

## Answer options
| Value | Label |
|-------|-------|
| Variable | Integer (mmHg) |

**Input format**: HealthKit-backed numeric integer field using `HKQuantityTypeIdentifierBloodPressureDiastolic`. Unit: mmHg. Required field.

## Observed values

**Total observations**: 3,703 — **type-enforced**: 3,703 (**unique**: 70) — raw Python types seen: `float` (3,703).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 30.00 |
| q25 | 70.00 |
| median | 76.00 |
| mean | 75.80 |
| q75 | 80.00 |
| max | 120 |
| std | 9.74 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `80.00` | 509 |
| `70.00` | 377 |
| `78.00` | 202 |
| `72.00` | 172 |
| `75.00` | 164 |
| `76.00` | 142 |
| `60.00` | 136 |
| `90.00` | 136 |
| `68.00` | 122 |
| `82.00` | 115 |
| `74.00` | 113 |
| `85.00` | 106 |
| `84.00` | 95 |
| `79.00` | 87 |
| `65.00` | 85 |
| `77.00` | 79 |
| `73.00` | 77 |
| `62.00` | 74 |
| `69.00` | 72 |
| `71.00` | 68 |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update), `34c0781` (MHC-178 identifier fix)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Diastolic BP is collected as part of the blood pressure pair but is **NOT directly used in the Framingham 10-year risk or heart age calculation**.
- Only systolic BP (diastolic's complement) appears in the Framingham coefficients.
- Presented alongside systolic BP on the "Blood pressure" form step with descriptive text: "Blood pressure (typically shown as systolic over diastolic)" (line 299).
- The HealthKit integration (line 315–317) shows the field is backed by HealthKit, suggesting it may be auto-populated if available.
- Marked as required (though it could be zero if not measured), appearing in the same form step as systolic BP for user convenience.
- Likely collected for research and completeness, as diastolic BP is relevant to overall cardiovascular health assessment even if not used in this specific Framingham risk model.
