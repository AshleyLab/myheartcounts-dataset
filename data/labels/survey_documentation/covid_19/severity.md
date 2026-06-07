# severity

**Benchmark column**: `field_severity`
**Raw identifier**: `severity`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~379 (main), ~379 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
Main survey: > When you felt the worst in the past month, from 0 (most sick) - 10 (perfect health) how did you feel?

Recurrent survey: > When you felt the worst in the past 2 weeks, from 0 (most sick) - 10 (perfect health) how did you feel?

## Answer options
Integer slider from 0 to 10, step 1.

| Value | Meaning |
|-------|---------|
| 0 | Most sick |
| 5 | Moderate |
| 10 | Perfect health |

## Observed values

**Total observations**: 174 — **type-enforced**: 174 (**unique**: 11) — raw Python types seen: `float` (174).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `8` | 40 | 23.0% |
| `7` | 21 | 12.1% |
| `6` | 20 | 11.5% |
| `3` | 18 | 10.3% |
| `9` | 18 | 10.3% |
| `10` | 14 | 8.0% |
| `4` | 12 | 6.9% |
| `2` | 10 | 5.7% |
| `5` | 10 | 5.7% |
| `0` | 7 | 4.0% |
| `1` | 4 | 2.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Visual analog scale measuring overall health severity; appears in both surveys with timeframe-specific wording

## Notes
Appears in both the main COVID survey and the recurrent COVID survey. Main survey references past month timeframe, while recurrent survey references past 2 weeks (matching bi-weekly schedule). Visual analog scale (0-10 slider) with reverse coding: lower values indicate worse health status.
