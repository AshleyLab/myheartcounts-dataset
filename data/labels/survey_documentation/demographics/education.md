# education

**Benchmark column**: `field_education`
**Raw identifier**: `education` (as in survey JSON)
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~332
- Survey identifier: `risk_factors_SchemaV2`

## Question
> What is the highest grade in school you finished? Choose the best answer.

## Answer options
| Value | Label |
|-------|-------|
| 1 | Didn't go to school |
| 2 | Grade school |
| 3 | High school diploma or G.E.D. |
| 4 | Some college or vocational school or Associate Degree |
| 5 | College graduate or Baccalaureate Degree |
| 6 | Master's Degree |
| 7 | Doctoral Degree (Ph.D., M.D., J.D., etc.) |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: false

## Observed values

**Total observations**: 11,677 — **type-enforced**: 11,677 (**unique**: 7) — raw Python types seen: `float` (11,677).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `5` | 4,152 | 35.6% |
| `4` | 2,643 | 22.6% |
| `6` | 2,542 | 21.8% |
| `7` | 1,329 | 11.4% |
| `3` | 811 | 6.9% |
| `2` | 190 | 1.6% |
| `1` | 10 | 0.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to education identifier

## Notes
Context variable providing educational attainment level for cardiovascular health study participants.
