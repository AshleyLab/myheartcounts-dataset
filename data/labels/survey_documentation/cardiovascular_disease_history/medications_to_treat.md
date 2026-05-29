# medications_to_treat

**Benchmark column**: `field_medications_to_treat`
**Raw identifier**: `medications_to_treat` (as in survey JSON)
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~44
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Do you take medications to treat the following risk factors (indicate all that apply)

## Answer options
| Value | Label |
|-------|-------|
| 1 | To treat and lower cholesterol |
| 2 | To treat hypertension and lower blood pressure |
| 3 | To treat diabetes/pre-diabetes and lower blood sugar |
| 4 | None of the above |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values

**Total observations**: 30,024 — **type-enforced**: 30,024 (**unique**: 12) — raw Python types seen: `list` (30,024).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 12 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(4)` | 22,669 |
| `(2)` | 2,478 |
| `(1, 2)` | 1,904 |
| `(1)` | 1,816 |
| `(1, 2, 3)` | 528 |
| `(3)` | 255 |
| `(1, 3)` | 186 |
| `(2, 3)` | 162 |
| `(1, 4)` | 11 |
| `(2, 4)` | 7 |
| `(3, 4)` | 5 |
| `(1, 2, 3, 4)` | 3 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 4,448 |
| 2 |  | 5,082 |
| 3 |  | 1,139 |
| 4 |  | 22,695 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to medications_to_treat identifier

## Notes
Context variable capturing current medication use for cardiovascular and metabolic risk factors: hyperlipidemia, hypertension, and diabetes/pre-diabetes. Multi-select question.
