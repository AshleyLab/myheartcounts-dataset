# cannabisSmoking

**Benchmark column**: `field_cannabisSmoking`
**Raw identifier**: `cannabisSmoking`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~811
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Do you smoke cannabis or cannabis containing products?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes, currently |
| 2 | No, but I have in the past |
| 3 | No, I never have |

## Observed values

**Total observations**: 738 — **type-enforced**: 738 (**unique**: 3) — raw Python types seen: `float` (738).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 416 | 56.4% |
| `2` | 271 | 36.7% |
| `1` | 51 | 6.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Cannabis module added; gating logic captures current/past/never usage in single question

## Notes
Gating logic: If value is 3 ("No, I never have"), skip to `cannabisVaping`. If value is 2 ("No, but I have in the past"), skip to `pastCannabisSmoking`. If value is 1 ("Yes, currently"), continues to currentCannabisSmoking. Central branching point for cannabis smoking module.
