# severity_covid

**Benchmark column**: `field_severity_covid`
**Raw identifier**: `severity-covid`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~365 (main), ~365 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> When you felt the worst with COVID-19, from 0 (most sick) - 10 (perfect health) how did you feel?

## Answer options
Integer slider from 0 to 10, step 1.

| Value | Meaning |
|-------|---------|
| 0 | Most sick |
| 5 | Moderate |
| 10 | Perfect health |

## Observed values

**Total observations**: 132 — **type-enforced**: 132 (**unique**: 11) — raw Python types seen: `float` (132).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 21 | 15.9% |
| `2` | 16 | 12.1% |
| `0` | 15 | 11.4% |
| `4` | 15 | 11.4% |
| `5` | 12 | 9.1% |
| `7` | 12 | 9.1% |
| `6` | 11 | 8.3% |
| `8` | 9 | 6.8% |
| `1` | 7 | 5.3% |
| `9` | 7 | 5.3% |
| `10` | 7 | 5.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Visual analog scale measuring worst COVID-19 severity; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Visual analog scale (0-10 slider) with reverse coding: lower values indicate worse health status. Captures subjective worst-case severity during COVID-19 illness.
