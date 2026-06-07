# symptoms_past_week

**Benchmark column**: `field_symptoms_past_week`
**Raw identifier**: `symptoms_past_week` (main survey) / `symptoms_past_2_weeks` (recurrent survey)
**Role**: context
**Type**: multi_categorical (source is `list<double>`; "Select all that apply")

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` (line ~156) and `cardio_covid_19_recurrent_survey.json` (line ~156)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
Main survey: > Did you experience any of the following symptoms in the past week ? (Select all that apply)

Recurrent survey: > Did you experience any of the following symptoms in the past 2 weeks ? (Select all that apply)

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

**Total observations**: 546 — **type-enforced**: 546 (**unique**: 42) — raw Python types seen: `list` (546).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 42 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(0)` | 458 |
| `(5)` | 16 |
| `(1)` | 10 |
| `(7)` | 6 |
| `(6)` | 4 |
| `(10)` | 4 |
| `(5, 10)` | 3 |
| `(8)` | 3 |
| `(3)` | 2 |
| `(3, 5, 6)` | 2 |
| `(3, 5)` | 2 |
| `(1, 7)` | 2 |
| `(8, 10)` | 2 |
| `(1, 3, 5, 7, 8)` | 2 |
| `(3, 5, 8)` | 2 |
| `(1, 3)` | 2 |
| `(2, 5)` | 1 |
| `(3, 4, 7, 10)` | 1 |
| `(1, 5, 8)` | 1 |
| `(5, 8)` | 1 |
| `(3, 4, 5, 6, 8, 9, 10, 11)` | 1 |
| `(3, 6, 10)` | 1 |
| `(1, 10)` | 1 |
| `(3, 5, 10)` | 1 |
| `(3, 6, 8, 11)` | 1 |
| `(2, 6)` | 1 |
| `(1, 5, 7)` | 1 |
| `(1, 2, 7, 8)` | 1 |
| `(5, 7, 8, 10, 11)` | 1 |
| `(6, 8)` | 1 |
| `(5, 7)` | 1 |
| `(2, 5, 9, 10, 11)` | 1 |
| `(11)` | 1 |
| `(5, 11)` | 1 |
| `(10, 11)` | 1 |
| `(3, 5, 6, 8, 11)` | 1 |
| `(1, 3, 5, 7, 8, 10, 11)` | 1 |
| `(1, 5)` | 1 |
| `(1, 6, 7, 8)` | 1 |
| `(5, 7, 10)` | 1 |
| `(3, 5, 9)` | 1 |
| `(1, 3, 5, 8)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 0 |  | 458 |
| 1 |  | 24 |
| 2 |  | 4 |
| 3 |  | 21 |
| 4 |  | 2 |
| 5 |  | 43 |
| 6 |  | 13 |
| 7 |  | 18 |
| 8 |  | 20 |
| 9 |  | 3 |
| 10 |  | 19 |
| 11 |  | 9 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Recent symptom assessment; multiple-select question; appears in both surveys with timeframe-specific wording

## Notes
Main survey identifier is `symptoms_past_week` (past 1 week). Recurrent survey identifier is `symptoms_past_2_weeks` (past 2 weeks) to match the bi-weekly survey schedule. Multiple-choice question allowing selection of multiple symptoms. Value 0 (None) has `ignoreOthers: true` flag, indicating it is mutually exclusive.
