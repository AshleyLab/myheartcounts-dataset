# race

**Benchmark column**: `field_race`
**Raw identifier**: `race` (as in survey JSON)
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~255
- Survey identifier: `risk_factors_SchemaV2`

## Question
> What is your race? Choose one or more races to indicate what you consider yourself to be.

## Answer options
| Value | Label |
|-------|-------|
| 1 | White |
| 2 | Black, African-American, or Negro |
| 3 | American Indian |
| 4 | Alaska Native |
| 5 | Asian Indian |
| 6 | Chinese |
| 7 | Filipino |
| 8 | Japanese |
| 9 | Korean |
| 10 | Vietnamese |
| 11 | Pacific islander |
| 12 | Some other race |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values

**Total observations**: 11,688 — **type-enforced**: 11,688 (**unique**: 73) — raw Python types seen: `list` (11,688).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 73 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(1)` | 9,425 |
| `(6)` | 456 |
| `(12)` | 385 |
| `(5)` | 375 |
| `(2)` | 353 |
| `(1, 3)` | 113 |
| `(7)` | 95 |
| `(1, 12)` | 81 |
| `(1, 2)` | 47 |
| `(9)` | 43 |
| `(8)` | 43 |
| `(3)` | 42 |
| `(10)` | 22 |
| `(1, 2, 3)` | 21 |
| `(1, 6)` | 20 |
| `(1, 8)` | 15 |
| `(2, 12)` | 11 |
| `(1, 7)` | 10 |
| `(11)` | 10 |
| `(1, 3, 12)` | 10 |
| `(1, 5)` | 8 |
| `(6, 12)` | 7 |
| `(5, 12)` | 7 |
| `(1, 9)` | 6 |
| `(1, 2, 12)` | 5 |
| `(4)` | 5 |
| `(1, 11)` | 4 |
| `(5, 6)` | 3 |
| `(1, 3, 8)` | 3 |
| `(1, 10)` | 3 |
| `(2, 3)` | 3 |
| `(1, 2, 3, 12)` | 3 |
| `(7, 11)` | 2 |
| `(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)` | 2 |
| `(6, 10)` | 2 |
| `(1, 6, 11)` | 2 |
| `(7, 9)` | 2 |
| `(11, 12)` | 2 |
| `(3, 12)` | 2 |
| `(7, 12)` | 2 |
| `(2, 9)` | 2 |
| `(2, 3, 12)` | 2 |
| `(1, 8, 12)` | 2 |
| `(1, 6, 9)` | 2 |
| `(1, 2, 6)` | 2 |
| `(2, 3, 5, 6)` | 1 |
| `(1, 7, 11, 12)` | 1 |
| `(5, 10)` | 1 |
| `(2, 7)` | 1 |
| `(1, 7, 9, 11)` | 1 |
| `(1, 6, 8)` | 1 |
| `(1, 2, 3, 6, 11)` | 1 |
| `(1, 7, 8, 9)` | 1 |
| `(1, 4, 12)` | 1 |
| `(1, 3, 7)` | 1 |
| `(7, 8, 11, 12)` | 1 |
| `(3, 4)` | 1 |
| `(5, 9)` | 1 |
| `(1, 7, 11)` | 1 |
| `(10, 11)` | 1 |
| `(2, 10)` | 1 |
| `(6, 7)` | 1 |
| `(2, 8)` | 1 |
| `(1, 6, 12)` | 1 |
| `(1, 2, 3, 5)` | 1 |
| `(2, 6)` | 1 |
| `(4, 7, 11)` | 1 |
| `(2, 3, 5)` | 1 |
| `(1, 6, 7)` | 1 |
| `(1, 2, 3, 5, 10, 12)` | 1 |
| `(3, 8)` | 1 |
| `(1, 4)` | 1 |
| `(1, 5, 12)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 9,798 |
| 2 |  | 460 |
| 3 |  | 209 |
| 4 |  | 11 |
| 5 |  | 402 |
| 6 |  | 503 |
| 7 |  | 123 |
| 8 |  | 70 |
| 9 |  | 60 |
| 10 |  | 33 |
| 11 |  | 29 |
| 12 |  | 527 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to race identifier

## Notes
Context variable providing self-identified racial/ethnic categories following standard demographic survey conventions. Multi-select question allowing respondents to indicate multiple races.
