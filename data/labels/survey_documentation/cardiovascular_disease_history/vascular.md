# vascular

> **Not in released benchmark.** Reason: broken multi-select shadow (Bug #149) — fanned out into Cerebrovascular Disease, PVD, PH targets. Removed at extraction, not via release_filter.. See `data/labels/RELEASE_NOTES.md` for the full disposition table.

**Benchmark column**: `vascular`
**Raw identifier**: `vascular` (as in survey JSON)
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~156
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Which vascular disease diagnosis have you received?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Stroke |
| 2 | Transient Ischemic Attack (TIA) |
| 3 | Carotid Artery Blockage/Stenosis |
| 4 | Carotid Artery Surgery or Stent |
| 5 | Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent) |
| 6 | Abdominal Aortic Aneurysm |
| 7 | None of the above |
| 8 | Pulmonary Arterial Hypertension |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values

**Total observations**: 0 — **type-enforced**: 0 (**unique**: 0) — raw Python types seen: —.
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

_Generated 2026-04-27 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `61c264a7…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: Most recent material change added Pulmonary Arterial Hypertension support

## Notes
Context variable capturing vascular disease diagnoses including cerebrovascular (stroke, TIA, carotid) and peripheral arterial disease. Multi-select question. Used as source for multiple target-level derived variables: Peripheral/Systemic Vascular Disease, Cerebrovascular Disease. Recent additions include Pulmonary Arterial Hypertension via MHC-780.
