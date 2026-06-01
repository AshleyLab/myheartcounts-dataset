# face_covering

**Benchmark column**: `field_face_covering`
**Raw identifier**: `face_covering`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~902
- Survey: `Covid_19_survey`

## Question
> Do you wear a face covering when you leave the house?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Always. |
| 2 | Most of the time. |
| 3 | Some of the time. |
| 4 | Never. |

## Observed values

**Total observations**: 1,027 — **type-enforced**: 1,027 (**unique**: 4) — raw Python types seen: `float` (1,027).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 473 | 46.1% |
| `2` | 284 | 27.7% |
| `3` | 181 | 17.6% |
| `4` | 89 | 8.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Face covering protective behavior ordinal scale; appears only in main survey

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Ordinal scale measuring frequency of face covering use when leaving home, ranging from always to never. Represents personal protective equipment (PPE) compliance behavior during COVID-19 pandemic.
