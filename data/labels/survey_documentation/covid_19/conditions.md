# conditions

**Benchmark column**: `field_conditions`
**Raw identifier**: `conditions`
**Role**: context
**Type**: multi_categorical (source is `list<double>`; "Select all that apply")

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~393
- Survey: `Covid_19_survey`

## Question
> What conditions do you have? (Select all that apply)

## Answer options
| Value | Label |
|-------|-------|
| 0 | None |
| 1 | Cardiovascular disease, including hypertension |
| 2 | Immunodeficiency, including HIV |
| 3 | Diabetes |
| 4 | Renal disease (ESRD on dialysis or not) |
| 5 | Liver disease |
| 6 | Chronic lung disease |
| 7 | Healthcare worker |
| 8 | Pregnancy or Post-partum (< 6 weeks) |

## Observed values

**Total observations**: 1,025 — **type-enforced**: 1,025 (**unique**: 37) — raw Python types seen: `list` (1,025).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 37 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(0)` | 595 |
| `(1)` | 186 |
| `(7)` | 86 |
| `(1, 3)` | 30 |
| `(1, 6)` | 20 |
| `(6)` | 19 |
| `(3)` | 15 |
| `(1, 7)` | 13 |
| `(2)` | 11 |
| `(5)` | 6 |
| `(1, 2)` | 6 |
| `(1, 4)` | 5 |
| `(1, 2, 3)` | 4 |
| `(6, 7)` | 3 |
| `(1, 5)` | 2 |
| `(1, 3, 5)` | 2 |
| `(1, 3, 4)` | 2 |
| `(1, 2, 4, 6)` | 1 |
| `(1, 3, 7)` | 1 |
| `(1, 6, 7)` | 1 |
| `(1, 5, 6)` | 1 |
| `(1, 2, 3, 5)` | 1 |
| `(1, 2, 7)` | 1 |
| `(1, 2, 3, 6, 7)` | 1 |
| `(2, 7)` | 1 |
| `(3, 6, 8)` | 1 |
| `(3, 5)` | 1 |
| `(1, 3, 4, 6)` | 1 |
| `(1, 4, 6)` | 1 |
| `(1, 2, 6, 7)` | 1 |
| `(5, 7)` | 1 |
| `(3, 6)` | 1 |
| `(2, 4)` | 1 |
| `(3, 5, 6)` | 1 |
| `(8)` | 1 |
| `(1, 3, 4, 5)` | 1 |
| `(2, 8)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 0 |  | 595 |
| 1 |  | 281 |
| 2 |  | 29 |
| 3 |  | 62 |
| 4 |  | 12 |
| 5 |  | 16 |
| 6 |  | 52 |
| 7 |  | 109 |
| 8 |  | 3 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Multiple comorbidities and occupational status assessment; multiple-select question

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Multiple-choice question allowing selection of multiple conditions. Value 0 (None) has `ignoreOthers: true` flag, indicating it is mutually exclusive. Conditional follow-up questions branch from this question for healthcare worker position (value 7) and immunodeficiency details (value 2).
