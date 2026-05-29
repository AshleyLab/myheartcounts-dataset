# healthcare_worker

**Benchmark column**: `field_healthcare_worker`
**Raw identifier**: `healthcare_worker`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~454
- Survey: `Covid_19_survey`

## Question
> What is your position?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Physician |
| 2 | Allied health provider |
| 3 | Nurse |
| 4 | Orderly |
| 5 | Technician |
| 6 | Food Worker |
| 7 | Maintenance staff |
| other | Other (freetext) |

## Observed values

**Total observations**: 111 — **type-enforced**: 111 (**unique**: 6) — raw Python types seen: `str` (111).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` (3) | 35 | 31.5% |
| `99` (Other) | 29 | 26.1% |
| `2` (2) | 19 | 17.1% |
| `1` (1) | 16 | 14.4% |
| `5` (5) | 11 | 9.9% |
| `7` (7) | 1 | 0.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Healthcare worker role categorization; conditional question to `conditions` value 7 (Healthcare worker)

## Notes
Appears only in the main COVID survey. Allows freetext entry via `allowOther: true`. Conditional follow-up question to healthcare worker selection in the `conditions` question.
