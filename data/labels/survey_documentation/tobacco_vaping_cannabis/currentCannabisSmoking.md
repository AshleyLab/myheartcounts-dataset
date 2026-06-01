# currentCannabisSmoking

**Benchmark column**: `field_currentCannabisSmoking`
**Raw identifier**: `currentCannabisSmoking`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~855
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> How often do you smoke cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | several times a day |
| 2 | 6-7 days a week |
| 3 | 3-5 days a week |
| 4 | 1-2 days a week |
| 5 | less than weekly |
| 6 | very seldom |

## Observed values

**Total observations**: 52 — **type-enforced**: 52 (**unique**: 6) — raw Python types seen: `float` (52).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 13 | 25.0% |
| `6` | 12 | 23.1% |
| `2` | 11 | 21.2% |
| `3` | 6 | 11.5% |
| `4` | 5 | 9.6% |
| `5` | 5 | 9.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Cannabis module added; frequency scale for current cannabis smoking

## Notes
Asked only if cannabisSmoking = 1 ("Yes, currently"). Ordinal frequency scale from daily to very seldom. No explicit gating rules defined in constraints.
