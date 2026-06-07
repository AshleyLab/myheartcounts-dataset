# most_intense_care

**Benchmark column**: `field_most_intense_care`
**Raw identifier**: `most_intense_care`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~244 (main), ~244 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> What was the most intense care you received for your symptoms?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Stayed home |
| 2 | Drive through testing without further care |
| 3 | Saw doctor or urgent care |
| 4 | Evaluated in Emergency Room |
| 5 | Admitted to hospital |

## Observed values

**Total observations**: 311 — **type-enforced**: 311 (**unique**: 5) — raw Python types seen: `float` (311).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 171 | 55.0% |
| `2` | 59 | 19.0% |
| `3` | 47 | 15.1% |
| `5` | 19 | 6.1% |
| `4` | 15 | 4.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Ordinal scale measuring healthcare utilization severity; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Ordinal scale reflecting escalating intensity of healthcare resource utilization, ranging from self-care to hospitalization.
