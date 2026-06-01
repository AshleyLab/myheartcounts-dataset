# lastCannabisVaping

**Benchmark column**: `field_lastCannabisVaping`
**Raw identifier**: `lastCannabisVaping`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: 1238
- Survey: Vaping and Smoking Survey

## Question
> When was the last time you vaped cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | < 1 week ago |
| 2 | 1 week to 1 month ago |
| 3 | 1 month to 6 months ago |
| 4 | 6 months to 2 years ago |
| 5 | 2 years to 5 years ago |
| 6 | > 5 years ago |

Data type: integer. `allowMultiple: false`, `allowOther: false`. UI hint: `list`.

## Observed values

**Total observations**: 94 — **type-enforced**: 94 (**unique**: 6) — raw Python types seen: `float` (94).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `4` | 37 | 39.4% |
| `5` | 23 | 24.5% |
| `6` | 13 | 13.8% |
| `3` | 12 | 12.8% |
| `2` | 7 | 7.4% |
| `1` | 2 | 2.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
Part of the Vaping and Smoking survey; see commit history in `cannabisVaping.md`.

## Notes
- Asked after `pastCannabisVaping` for participants who vaped cannabis in the past but not currently.
- Sibling `lastCannabisSmoking` (in `tobacco_vaping_cannabis/`) — same recency scale for cannabis smoking.
- Missed in the original documentation pass.
