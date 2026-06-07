# readinessQuitSmokeless

**Benchmark column**: `field_readinessQuitSmokeless`
**Raw identifier**: `readinessQuitSmokeless`
**Role**: context
**Type**: ordinal (1-10 slider)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~581
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> On a scale of 1-10, how ready do you feel to quit smokeless tobacco?

## Answer options
- **Range**: 1 to 10
- **Step**: 1
- **Unit**: Readiness scale (1 = not ready, 10 = very ready)
- **UI Hint**: slider

## Observed values

**Total observations**: 98 — **type-enforced**: 98 (**unique**: 10) — raw Python types seen: `float` (98).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `10` | 72 | 73.5% |
| `1` | 9 | 9.2% |
| `6` | 5 | 5.1% |
| `5` | 3 | 3.1% |
| `2` | 2 | 2.0% |
| `3` | 2 | 2.0% |
| `4` | 2 | 2.0% |
| `7` | 1 | 1.0% |
| `8` | 1 | 1.0% |
| `9` | 1 | 1.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Continuous readiness scale (1-10). No explicit gating logic; standard progression in smokeless tobacco module.
