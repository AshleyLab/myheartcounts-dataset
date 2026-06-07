# lastCannabisSmoking

**Benchmark column**: `field_lastCannabisSmoking`
**Raw identifier**: `lastCannabisSmoking`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~1001
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> When was the last time you smoked cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | < 1 week ago |
| 2 | 1 week to 1 month ago |
| 3 | 1 month to 6 months ago |
| 4 | 6 months to 2 years ago |
| 5 | 2 years to 5 years ago |
| 6 | > 5 years ago |

## Observed values

**Total observations**: 272 — **type-enforced**: 272 (**unique**: 5) — raw Python types seen: `float` (272).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `6` | 166 | 61.0% |
| `5` | 41 | 15.1% |
| `4` | 39 | 14.3% |
| `3` | 19 | 7.0% |
| `2` | 7 | 2.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Cannabis module added; temporal scale for recency of cannabis smoking

## Notes
Asked if cannabisSmoking = 2 ("No, but I have in the past") after durationCannabisSmoking (indicated by go rule in durationCannabisSmoking). Ordinal time interval scale from < 1 week to > 5 years. No explicit gating rules defined in constraints.
