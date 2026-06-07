# pastCannabisSmoking

**Benchmark column**: `field_pastCannabisSmoking`
**Raw identifier**: `pastCannabisSmoking`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~954
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> For how long did you smoke cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | < 1year |
| 2 | 1-5yrs |
| 3 | 6-10 yrs |
| 4 | 11-15 yrs |
| 5 | 16-20 yrs |
| 6 | >20 yrs |

## Observed values

**Total observations**: 271 — **type-enforced**: 271 (**unique**: 6) — raw Python types seen: `float` (271).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 137 | 50.6% |
| `2` | 90 | 33.2% |
| `6` | 20 | 7.4% |
| `3` | 16 | 5.9% |
| `4` | 5 | 1.8% |
| `5` | 3 | 1.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Cannabis module added; duration scale for past cannabis smoking

## Notes
Asked if cannabisSmoking = 2 ("No, but I have in the past"). Ordinal duration scale for historical cannabis smoking. No explicit gating rules defined in constraints.
