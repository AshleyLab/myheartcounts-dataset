# durationQuitSmokeless

**Benchmark column**: `field_durationQuitSmokeless`
**Raw identifier**: `durationQuitSmokeless`
**Role**: context
**Type**: categorical (codes 1-3 are ordered durations Days < Months < Years, but codes 4=Never and 5=Don't know break the ordering, so the field is declared categorical; filter to {1, 2, 3} for ordinal analysis)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~614
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> What is the longest period that you were able to chewing tobacco (chewing tobacco, snuff, snus, and dissolvable tobacco products)?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Days |
| 2 | Months |
| 3 | Years |
| 4 | Never |
| 5 | Don't know |

## Observed values

**Total observations**: 35 — **type-enforced**: 35 (**unique**: 5) — raw Python types seen: `float` (35).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 14 | 40.0% |
| `5` | 9 | 25.7% |
| `1` | 4 | 11.4% |
| `2` | 4 | 11.4% |
| `4` | 4 | 11.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Asked only if everQuitSmokeless is true (Yes). Ordinal scale of time periods for longest quit duration. Note: prompt has minor grammar issue ("to chewing tobacco" instead of "at quitting").
