# family_history

**Benchmark column**: `field_family_history`
**Raw identifier**: `family_history` (as in survey JSON)
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~10
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Do you have a family history of early heart disease?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Father or brother with heart attack or coronary artery disease before age 55 |
| 2 | Mother or sister with heart attack or coronary artery disease before age 65 |
| 3 | None of the above |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values

**Total observations**: 29,867 — **type-enforced**: 29,867 (**unique**: 7) — raw Python types seen: `list` (29,867).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 7 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(3)` | 23,020 |
| `(1)` | 4,533 |
| `(2)` | 1,578 |
| `(1, 2)` | 671 |
| `(1, 3)` | 49 |
| `(2, 3)` | 13 |
| `(1, 2, 3)` | 3 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 5,256 |
| 2 |  | 2,265 |
| 3 |  | 23,085 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to family_history identifier

## Notes
Context variable capturing early heart disease family history with sex and age-stratified thresholds per cardiovascular risk guidelines. Multi-select question.
