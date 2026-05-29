# symptoms_week_preceding

**Benchmark column**: `field_symptoms_week_preceding`
**Raw identifier**: `symptoms_week_preceding`
**Role**: context
**Type**: multi_categorical (source is `list<double>`; "Select all that apply")

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~80 (main), ~80 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> Did you experience any of the following symptoms in the week preceding your COVID test? (Select all that apply)

## Answer options
| Value | Label |
|-------|-------|
| 0 | None |
| 1 | Cough |
| 2 | Fever |
| 3 | Shortness of breath |
| 4 | Loss of smell / taste |
| 5 | Fatigue / lethargy |
| 6 | Chest pain |
| 7 | Sore throat |
| 8 | Muscle aches / muscle pain |
| 9 | Anorexia, loss of appetite |
| 10 | Diarrhea |
| 11 | Nausea/vomiting |

## Observed values

**Total observations**: 490 — **type-enforced**: 490 (**unique**: 134) — raw Python types seen: `list` (490).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 134 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(0)` | 290 |
| `(7)` | 11 |
| `(1)` | 7 |
| `(2)` | 5 |
| `(1, 7)` | 5 |
| `(2, 5, 8)` | 5 |
| `(5)` | 4 |
| `(10)` | 4 |
| `(1, 2, 3, 4, 5, 6, 7, 8, 9)` | 4 |
| `(1, 5)` | 3 |
| `(1, 3)` | 3 |
| `(2, 5, 7, 8)` | 3 |
| `(1, 2, 7)` | 3 |
| `(5, 8)` | 3 |
| `(1, 2, 3, 4, 5, 6, 8)` | 3 |
| `(1, 5, 7, 8)` | 3 |
| `(1, 7, 10)` | 2 |
| `(2, 5, 7)` | 2 |
| `(1, 7, 8)` | 2 |
| `(5, 7)` | 2 |
| `(1, 2, 3, 4, 5, 7, 8, 9, 10)` | 2 |
| `(1, 2, 3, 4, 5, 6, 8, 9, 11)` | 2 |
| `(1, 2, 3, 4, 5, 9, 10, 11)` | 2 |
| `(1, 2)` | 2 |
| `(1, 2, 3, 4, 5, 8)` | 2 |
| `(1, 2, 3, 4, 5, 7, 8, 10)` | 2 |
| `(4)` | 2 |
| `(1, 2, 3, 4, 5, 8, 9)` | 2 |
| `(1, 5, 7, 8, 9)` | 2 |
| `(1, 2, 5, 8, 9, 10)` | 2 |
| `(1, 3, 5, 7)` | 2 |
| `(5, 7, 8, 10)` | 2 |
| `(2, 6)` | 1 |
| `(4, 5, 7, 8, 9, 10)` | 1 |
| `(1, 2, 4, 5, 7, 8)` | 1 |
| `(2, 5)` | 1 |
| `(2, 8)` | 1 |
| `(2, 10)` | 1 |
| `(1, 2, 3, 4, 5, 7, 8, 9, 11)` | 1 |
| `(1, 3, 9, 11)` | 1 |
| `(5, 8, 9, 10, 11)` | 1 |
| `(5, 10)` | 1 |
| `(1, 3, 5, 8)` | 1 |
| `(1, 4, 5, 6, 7, 8)` | 1 |
| `(1, 2, 5, 8)` | 1 |
| `(1, 2, 3, 4, 5, 6, 8, 9, 10)` | 1 |
| `(2, 4)` | 1 |
| `(6)` | 1 |
| `(2, 4, 7, 8)` | 1 |
| `(3, 4, 5, 7, 10)` | 1 |
| `(4, 5, 7, 8, 9, 11)` | 1 |
| `(4, 5, 8)` | 1 |
| `(7, 10)` | 1 |
| `(1, 2, 5, 7, 8)` | 1 |
| `(1, 3, 5, 6, 7)` | 1 |
| `(1, 2, 3, 4, 5, 6)` | 1 |
| `(11)` | 1 |
| `(1, 4)` | 1 |
| `(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)` | 1 |
| `(3, 5, 7, 8, 10, 11)` | 1 |
| `(1, 5, 8)` | 1 |
| `(1, 2, 3, 4, 5, 6, 7, 8, 9, 11)` | 1 |
| `(1, 7, 11)` | 1 |
| `(5, 7, 11)` | 1 |
| `(2, 4, 8)` | 1 |
| `(5, 6, 7, 8, 9, 10, 11)` | 1 |
| `(1, 3, 4, 5, 6, 7, 8)` | 1 |
| `(1, 3, 4, 5, 6, 7, 8, 10)` | 1 |
| `(6, 7, 8)` | 1 |
| `(2, 3, 5, 10)` | 1 |
| `(2, 4, 5)` | 1 |
| `(1, 2, 4, 5, 8, 9, 11)` | 1 |
| `(1, 7, 8, 10)` | 1 |
| `(1, 3, 5, 6, 7, 8)` | 1 |
| `(5, 7, 9)` | 1 |
| `(1, 2, 3, 5, 6, 8, 9, 10)` | 1 |
| `(1, 2, 5)` | 1 |
| `(8)` | 1 |
| `(1, 4, 5, 7, 8)` | 1 |
| `(1, 2, 3, 5, 8, 9, 11)` | 1 |
| `(1, 4, 7)` | 1 |
| `(2, 4, 5, 10)` | 1 |
| `(4, 5, 7, 8)` | 1 |
| `(1, 3, 4, 5, 8)` | 1 |
| `(8, 10, 11)` | 1 |
| `(1, 2, 3, 4)` | 1 |
| `(1, 2, 3, 6, 7, 10)` | 1 |
| `(5, 9)` | 1 |
| `(3, 4, 5, 6, 7, 8)` | 1 |
| `(10, 11)` | 1 |
| `(1, 4, 5, 11)` | 1 |
| `(1, 3, 7, 10, 11)` | 1 |
| `(1, 4, 5, 7)` | 1 |
| `(1, 3, 4, 5, 8, 9)` | 1 |
| `(1, 2, 6, 8)` | 1 |
| `(1, 3, 4, 5, 6, 7, 8, 9, 10, 11)` | 1 |
| `(3, 5, 8, 9)` | 1 |
| `(1, 2, 3, 5, 7)` | 1 |
| `(1, 2, 4)` | 1 |
| `(1, 2, 3, 5, 7, 9, 11)` | 1 |
| `(8, 10)` | 1 |
| `(1, 4, 5, 8)` | 1 |
| `(1, 2, 3, 5)` | 1 |
| `(1, 3, 5, 7, 8, 11)` | 1 |
| `(1, 2, 4, 8)` | 1 |
| `(5, 8, 9)` | 1 |
| `(2, 5, 8, 10)` | 1 |
| `(1, 2, 3, 5, 7, 8, 10)` | 1 |
| `(3, 5)` | 1 |
| `(2, 3, 6, 7, 11)` | 1 |
| `(2, 4, 5, 6, 7, 8, 9, 10, 11)` | 1 |
| `(3, 5, 6, 8)` | 1 |
| `(3, 4, 5, 6, 8, 10, 11)` | 1 |
| `(7, 8, 10)` | 1 |
| `(1, 5, 7, 11)` | 1 |
| `(1, 2, 6, 10, 11)` | 1 |
| `(1, 3, 5)` | 1 |
| `(1, 5, 6, 7)` | 1 |
| `(1, 2, 3, 5, 7, 8, 9, 10)` | 1 |
| `(1, 2, 5, 7)` | 1 |
| `(1, 5, 7)` | 1 |
| `(4, 5)` | 1 |
| `(1, 2, 3, 4, 8)` | 1 |
| `(1, 2, 4, 7, 10)` | 1 |
| `(2, 5, 8, 10, 11)` | 1 |
| `(1, 4, 7, 8)` | 1 |
| `(1, 2, 3, 4, 5, 7, 8, 9, 10, 11)` | 1 |
| `(1, 3, 5, 7, 8, 9, 11)` | 1 |
| `(1, 2, 3, 5, 8)` | 1 |
| `(1, 2, 5, 7, 9, 10)` | 1 |
| `(1, 2, 10)` | 1 |
| `(2, 3, 7, 8)` | 1 |
| `(1, 2, 4, 5, 8, 9, 10, 11)` | 1 |
| `(1, 2, 3, 5, 7, 8, 11)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 0 |  | 290 |
| 1 |  | 114 |
| 2 |  | 88 |
| 3 |  | 65 |
| 4 |  | 62 |
| 5 |  | 121 |
| 6 |  | 33 |
| 7 |  | 96 |
| 8 |  | 97 |
| 9 |  | 41 |
| 10 |  | 50 |
| 11 |  | 34 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Symptom assessment preceding COVID test; multiple-select question; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Multiple-choice question allowing selection of multiple symptoms. Value 0 (None) has `ignoreOthers: true` flag, indicating it is mutually exclusive.
