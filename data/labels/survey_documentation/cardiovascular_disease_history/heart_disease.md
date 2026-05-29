# heart_disease

> **Not in released benchmark.** Reason: broken multi-select shadow (Bug #149) — fanned out into per-subtype binary targets (CAD, Afib, CHF, PH, Congenital Heart). Removed at extraction, not via release_filter.. See `data/labels/RELEASE_NOTES.md` for the full disposition table.

**Benchmark column**: `heart_disease`
**Raw identifier**: `heart_disease` (as in survey JSON)
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~83
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Have you been diagnosed with any of the below diseases?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Heart Attack/Myocardial Infarction |
| 2 | Heart Bypass Surgery |
| 3 | Coronary Blockage/Stenosis |
| 4 | Coronary Stent/Angioplasty |
| 5 | Angina (heart-related chest pains) |
| 6 | High Coronary Calcium Score |
| 7 | Heart Failure or CHF |
| 8 | Atrial fibrillation (Afib) |
| 9 | Congenital Heart Defect |
| 10 | None of the above |
| 11 | Pulmonary Hypertension |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values

**Total observations**: 0 — **type-enforced**: 0 (**unique**: 0) — raw Python types seen: —.
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

_Generated 2026-04-27 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `61c264a7…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: a8e47e1 (2024) MHC-734 CVHealth Survey - add PH Support
- Notes: Most recent material change added Pulmonary Hypertension (value 11) support

## Notes
Context variable capturing cardiac disease diagnoses. Multi-select question. Used as source for multiple target-level derived variables: cardiovascular_disease, Heart Failure or CHF, Afib, CAD, Congenital Heart, PH. Recent additions include Pulmonary Hypertension support via MHC-734.
