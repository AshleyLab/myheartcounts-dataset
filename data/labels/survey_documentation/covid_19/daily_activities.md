# daily_activities

**Benchmark column**: `field_daily_activities`
**Raw identifier**: `daily_activities`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~284 (main), ~284 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> While symptomatic, could you complete your usual daily activities?

## Answer options
| Value | Label |
|-------|-------|
| 1 | None |
| 2 | Some |
| 3 | Most |
| 4 | All |

## Observed values

**Total observations**: 291 — **type-enforced**: 291 (**unique**: 4) — raw Python types seen: `float` (291).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `4` | 113 | 38.8% |
| `2` | 94 | 32.3% |
| `3` | 55 | 18.9% |
| `1` | 29 | 10.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Functional capacity ordinal scale; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Ordinal scale measuring functional capacity/activity limitation during COVID-19 symptomatic period, ranging from complete functional impairment (None) to no impairment (All).
