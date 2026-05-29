# durationQuitVaping

**Benchmark column**: `field_durationQuitVaping`
**Raw identifier**: `durationQuitVaping`
**Role**: context
**Type**: categorical (codes 1-3 are ordered durations Days < Months < Years, but codes 4=Never and 5=Don't know break the ordering, so the field is declared categorical; filter to {1, 2, 3} for ordinal analysis)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~155
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> What is the longest period that you were able to quit vaping?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Days |
| 2 | Months |
| 3 | Years |
| 4 | Never |
| 5 | Don't know |

## Observed values

**Total observations**: 83 — **type-enforced**: 83 (**unique**: 5) — raw Python types seen: `float` (83).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 35 | 42.2% |
| `2` | 31 | 37.3% |
| `1` | 15 | 18.1% |
| `4` | 1 | 1.2% |
| `5` | 1 | 1.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original vaping survey

## Notes
Asked only if everQuitVaping is true (Yes). Ordinal scale of time periods for longest quit duration.
