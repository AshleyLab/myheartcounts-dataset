# covid_relatives

> **Not in released benchmark.** Reason: rare-cell — single-digit positive counts under joint queries. See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `covid_relatives`
**Raw identifier**: `relatives`
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~790
- Survey: `Covid_19_survey`

## Question
> Do you have any relatives related by blood who have tested positive for COVID-19?

## Answer options
| Value | Label |
|-------|-------|
| true | Yes |
| false | No |

## Observed values

**Total observations**: 737 — **type-enforced**: 737 (**unique**: 2) — raw Python types seen: `bool` (737).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 584 | 79.2% |
| `True` | 153 | 20.8% |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Part of core COVID survey; stable since initial COVID survey implementation

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Conditional question follow-up `relatives_details` collects relationship information.
