# ethnicity

**Benchmark column**: `field_ethnicity`
**Raw identifier**: `ethnicity` (as in survey JSON)
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~214
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Are you Spanish/Hispanic/Latino? Choose the best answer.

## Answer options
| Value | Label |
|-------|-------|
| 1 | No, not Spanish/Hispanic/Latino |
| 2 | Yes, Puerto Rican |
| 3 | Yes, Mexican, Mexican American, or Chicano |
| 4 | Yes, Cuban |
| 5 | Yes, other Spanish/Hispanic/Latina |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: false

## Observed values

**Total observations**: 11,636 — **type-enforced**: 11,636 (**unique**: 5) — raw Python types seen: `float` (11,636).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 10,700 | 92.0% |
| `5` | 429 | 3.7% |
| `3` | 317 | 2.7% |
| `2` | 118 | 1.0% |
| `4` | 72 | 0.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to ethnicity identifier

## Notes
Context variable providing demographic ethnicity/Hispanic origin information for cardiovascular health study participants.
