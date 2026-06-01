# readinessQuitVaping

**Benchmark column**: `field_readinessQuitVaping`
**Raw identifier**: `readinessQuitVaping`
**Role**: context
**Type**: ordinal (1-10 slider)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~57
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> On a scale of 1-10, how ready do you feel to quit vaping?

## Answer options
- **Range**: 1 to 10
- **Step**: 1
- **Unit**: Readiness scale (1 = not ready, 10 = very ready)
- **UI Hint**: slider

## Observed values

**Total observations**: 93 — **type-enforced**: 93 (**unique**: 10) — raw Python types seen: `float` (93).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 19 | 20.4% |
| `5` | 13 | 14.0% |
| `3` | 11 | 11.8% |
| `2` | 10 | 10.8% |
| `9` | 10 | 10.8% |
| `4` | 8 | 8.6% |
| `8` | 7 | 7.5% |
| `10` | 6 | 6.5% |
| `7` | 5 | 5.4% |
| `6` | 4 | 4.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: No targeted changes to readinessQuitVaping identifier; survey structure stable

## Notes
Continuous readiness scale (1-10). No visible branching logic in the current survey definition.
