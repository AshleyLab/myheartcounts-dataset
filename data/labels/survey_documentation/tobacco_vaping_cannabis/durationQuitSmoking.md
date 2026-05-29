# durationQuitSmoking

**Benchmark column**: `field_durationQuitSmoking`
**Raw identifier**: `durationQuitSmoking`
**Role**: context
**Type**: categorical (codes 1-3 are ordered durations Days < Months < Years, but codes 4=Never and 5=Don't know break the ordering, so the field is declared categorical; filter to {1, 2, 3} for ordinal analysis)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~387
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> What is the longest period that you were able to quit smoking cigarettes?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Days |
| 2 | Months |
| 3 | Years |
| 4 | Never |
| 5 | Don't know |

## Observed values

**Total observations**: 132 — **type-enforced**: 132 (**unique**: 5) — raw Python types seen: `float` (132).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 85 | 64.4% |
| `2` | 34 | 25.8% |
| `1` | 10 | 7.6% |
| `4` | 2 | 1.5% |
| `5` | 1 | 0.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original smoking survey

## Notes
Asked only if everQuitSmoking is true (Yes). Ordinal scale of time periods for longest quit duration.
