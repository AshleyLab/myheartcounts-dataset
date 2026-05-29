# readinessQuitSmoking

**Benchmark column**: `field_readinessQuitSmoking`
**Raw identifier**: `readinessQuitSmoking`
**Role**: context
**Type**: ordinal (1-10 slider)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~289
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> On a scale of 1-10, how ready do you feel to quit smoking cigarettes?

## Answer options
- **Range**: 1 to 10
- **Step**: 1
- **Unit**: Readiness scale (1 = not ready, 10 = very ready)
- **UI Hint**: slider

## Observed values

**Total observations**: 95 — **type-enforced**: 95 (**unique**: 10) — raw Python types seen: `float` (95).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `10` | 15 | 15.8% |
| `1` | 14 | 14.7% |
| `5` | 14 | 14.7% |
| `9` | 11 | 11.6% |
| `8` | 9 | 9.5% |
| `6` | 8 | 8.4% |
| `2` | 7 | 7.4% |
| `7` | 7 | 7.4% |
| `3` | 5 | 5.3% |
| `4` | 5 | 5.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; survey structure unchanged

## Notes
Continuous readiness scale (1-10). No explicit gating logic; standard progression in smoking module.
