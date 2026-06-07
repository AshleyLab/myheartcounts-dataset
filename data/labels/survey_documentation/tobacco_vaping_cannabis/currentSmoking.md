# currentSmoking

**Benchmark column**: `field_currentSmoking`
**Raw identifier**: `currentSmoking`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~240
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Do you currently smoke cigarettes?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Daily |
| 2 | Less than daily |
| 3 | Not at all |
| 4 | Don't know |

## Observed values

**Total observations**: 1,330 — **type-enforced**: 1,330 (**unique**: 4) — raw Python types seen: `float` (1,330).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 1,239 | 93.2% |
| `1` | 64 | 4.8% |
| `2` | 24 | 1.8% |
| `4` | 3 | 0.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original survey

## Notes
Gating logic: this is the central branching point for the smoking module — depending on the response, the iOS survey routes to either continued smoking detail or the smokeless-tobacco section. (The lifetime-cigarette follow-up that immediately follows in the raw survey is not exposed in the benchmark — it had zero observations and was dropped from the API.)
