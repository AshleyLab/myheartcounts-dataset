# days_admitted

**Benchmark column**: `field_days_admitted`
**Raw identifier**: `days_admitted`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~319 (main), ~319 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> For how many days were you admitted?

## Answer options
Integer input field. Step: 1. Minimum: 0 (implicit). No explicit maximum.

## Observed values

**Total observations**: 19 — **type-enforced**: 19 (**unique**: 11) — raw Python types seen: `float` (19).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 2.00 |
| q25 | 3.00 |
| median | 4.00 |
| mean | 9.26 |
| q75 | 8.00 |
| max | 70.00 |
| std | 15.51 |

**Top 11 most frequent values**:

| value | count |
|------:|------:|
| `2.00` | 4 |
| `3.00` | 4 |
| `4.00` | 2 |
| `5.00` | 2 |
| `6.00` | 1 |
| `7.00` | 1 |
| `9.00` | 1 |
| `11.00` | 1 |
| `14.00` | 1 |
| `21.00` | 1 |
| `70.00` | 1 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Hospital admission duration measure; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Numeric field capturing duration of hospitalization in days.
