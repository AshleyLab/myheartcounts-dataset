# covid

**Benchmark column**: `field_covid`
**Raw identifier**: `covid`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~15 (main), ~15 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
Main survey: > Have you ever had RNA testing for current COVID-19?

Recurrent survey: > Have you had RNA testing for current COVID-19 in the past 2 weeks?

## Answer options
| Value | Label |
|-------|-------|
| 1 | No |
| 2 | Yes, it was positive |
| 3 | Yes, it was negative |
| 4 | Yes, I am awaiting results |

## Observed values

**Total observations**: 1,029 — **type-enforced**: 1,029 (**unique**: 4) — raw Python types seen: `float` (1,029).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 625 | 60.7% |
| `3` | 289 | 28.1% |
| `2` | 104 | 10.1% |
| `4` | 11 | 1.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Core COVID-19 test result assessment; appears in both surveys with timeframe-specific wording

## Notes
Appears in both the main COVID survey and the recurrent COVID survey. The main survey assesses lifetime testing history, while the recurrent survey captures bi-weekly testing history (past 2 weeks). Ordinal scale reflecting increasing certainty of COVID-19 status.
