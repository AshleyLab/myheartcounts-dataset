# icu_treated

> **Not in released benchmark.** Reason: rare-cell + clinical specificity — single-digit positive counts. See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `icu_treated`
**Raw identifier**: `icu_treated`
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~331 (main), ~331 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
> Were you treated in the ICU?

## Answer options
| Value | Label |
|-------|-------|
| true | Yes |
| false | No |

## Observed values

**Total observations**: 13 — **type-enforced**: 13 (**unique**: 2) — raw Python types seen: `bool` (13).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `True` | 8 | 61.5% |
| `False` | 5 | 38.5% |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Part of hospitalization severity assessment; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey with identical structure. Conditional follow-up question `days_icu` collects duration information.
