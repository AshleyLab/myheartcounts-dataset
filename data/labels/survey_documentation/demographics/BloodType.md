# BloodType

> **Not in released benchmark.** Reason: PHI — rare ABO/Rh combinations are uniquely identifying. See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `BloodType`
**Raw identifier**: `blood_type`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~720
- Survey: `Covid_19_survey`

## Question
> What is your blood type?

## Answer options
| Value | Label |
|-------|-------|
| 1 | A |
| 2 | B |
| 3 | AB |
| 4 | O |
| 5 | I don't know |

## Observed values

**Total observations**: 476 — **type-enforced**: 0 (**unique**: 0) — raw Python types seen: `str` (476).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 476 dictionary-miss (`KeyError`).

**Raw stored values (top 8)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `O+` | 172 |
| `A+` | 153 |
| `B+` | 45 |
| `O-` | 39 |
| `A-` | 36 |
| `AB+` | 19 |
| `B-` | 8 |
| `AB-` | 4 |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Blood type genetic marker for COVID-19 severity association research

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Includes "I don't know" option (value 5) for participants unable to report blood type.
