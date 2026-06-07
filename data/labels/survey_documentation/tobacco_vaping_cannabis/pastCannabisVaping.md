# pastCannabisVaping

**Benchmark column**: `field_pastCannabisVaping`
**Raw identifier**: `pastCannabisVaping`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: 1191
- Survey: Vaping and Smoking Survey

## Question
> For how long did you vape cannabis?

## Answer options
| Value | Label |
|-------|-------|
| 1 | < 1 year |
| 2 | 1-5 yrs |
| 3 | 6-10 yrs |
| 4 | 11-15 yrs |
| 5 | 16-20 yrs |
| 6 | > 20 yrs |

Data type: integer. `allowMultiple: false`, `allowOther: false`. UI hint: `list`.

## Observed values

**Total observations**: 95 — **type-enforced**: 95 (**unique**: 4) — raw Python types seen: `float` (95).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 67 | 70.5% |
| `2` | 22 | 23.2% |
| `3` | 3 | 3.2% |
| `5` | 3 | 3.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
Part of the Vaping and Smoking survey; see commit history in `cannabisVaping.md`.

## Notes
- Asked when `cannabisVaping == 2` (vaped cannabis in the past but not currently).
- Sibling `durationCannabisVaping` (line 1139, same file) asks the same duration question in present tense for *current* vapers. Not in the benchmark spec but structurally identical.
- Sibling `pastCannabisSmoking` (in `tobacco_vaping_cannabis/`) — same duration scale for past cannabis *smoking*.
- Missed in the original documentation pass.
