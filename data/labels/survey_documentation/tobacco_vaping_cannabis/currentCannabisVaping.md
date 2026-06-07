# currentCannabisVaping

**Benchmark column**: `field_currentCannabisVaping`
**Raw identifier**: `currentCannabisVaping`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: 1092
- Survey: Vaping and Smoking Survey

## Question
> How often do you vape cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | several times a day |
| 2 | 6-7 days a week |
| 3 | 3-5 days a week |
| 4 | 1-2 days a week |
| 5 | less than weekly |
| 6 | very seldom |

Data type: integer. `allowMultiple: false`, `allowOther: false`. UI hint: `list`.

## Observed values

**Total observations**: 30 — **type-enforced**: 30 (**unique**: 6) — raw Python types seen: `float` (30).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 8 | 26.7% |
| `6` | 6 | 20.0% |
| `2` | 5 | 16.7% |
| `4` | 5 | 16.7% |
| `1` | 3 | 10.0% |
| `5` | 3 | 10.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
Part of the Vaping and Smoking survey; see commit history in `cannabisVaping.md` (same file).

## Notes
- Only asked if `cannabisVaping == 1` (currently vaping cannabis).
- Sibling of `currentCannabisSmoking` (in `tobacco_vaping_cannabis/`) — same frequency scale but for smoking cannabis rather than vaping it.
- Missed in the original documentation pass.
