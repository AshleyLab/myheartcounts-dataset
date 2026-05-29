# covid_serologic

**Benchmark column**: `field_covid_serologic`
**Raw identifier**: `covid_serologic`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json` and `cardio_covid_19_recurrent_survey.json`
- Line: ~50 (main), ~50 (recurrent)
- Survey: `Covid_19_survey` / `Covid_19_recurrent_survey`

## Question
Main survey: > Have you ever had serologic or antibody testing for COVID-19?

Recurrent survey: > Have you had serologic or antibody testing) (blood test) for COVID-19 in the past 2 weeks

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes, it was positive (IgM(+) and/or IgG(+)) |
| 2 | Yes, it was negative (IgM(-) & IgG(-)) |
| 3 | No |

## Observed values

**Total observations**: 1,026 — **type-enforced**: 1,026 (**unique**: 3) — raw Python types seen: `float` (1,026).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 759 | 74.0% |
| `2` | 184 | 17.9% |
| `1` | 83 | 8.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20 (main survey), 4 (recurrent survey)
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Serologic/antibody testing for COVID-19 immunity assessment; appears in both surveys

## Notes
Appears in both the main COVID survey and the recurrent COVID survey. Tests for immune response to COVID-19 infection via antibodies. Value ordering differs between positive/negative vs. "No testing" for data analysis purposes.
