# exposure

**Benchmark column**: `field_exposure`
**Raw identifier**: `exposure`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~760
- Survey: `Covid_19_survey`

## Question
> Have you been exposed to anyone that tested positive for COVID-19?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes. I live with someone who has COVID-19 and was exposed to him/her. |
| 2 | Yes. I was directly exposed to someone with COVID-19 outside of my own home. |
| 3 | No |

## Observed values

**Total observations**: 1,028 — **type-enforced**: 1,028 (**unique**: 3) — raw Python types seen: `float` (1,028).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 829 | 80.6% |
| `2` | 136 | 13.2% |
| `1` | 63 | 6.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: COVID-19 exposure context; ordinal scale with intensity of exposure

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Ordinal scale reflecting intensity/proximity of COVID-19 exposure, with household exposure as most intense, followed by direct external exposure, followed by no exposure.
