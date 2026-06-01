# self_isolating

**Benchmark column**: `field_self_isolating`
**Raw identifier**: `self_isolating`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~866
- Survey: `Covid_19_survey`

## Question
> To what extent are you currently self-isolating?

Note: Self-isolation defined as avoiding people and situations that may lead to COVID-19 exposure, including avoiding leaving home for school, work, social events, community gatherings, sporting events, cultural events, and religious gatherings.

## Answer options
| Value | Label |
|-------|-------|
| 1 | Always. I almost never leave home. |
| 2 | Most of the time. I leave the home very infrequently and only for essential activities, like shopping for food. |
| 3 | Some of the time. I still go out sometimes but leave home less than I did before. |
| 4 | Never. I have not changed my routine and go out just as frequently as I used to. |

## Observed values

**Total observations**: 1,018 — **type-enforced**: 1,018 (**unique**: 4) — raw Python types seen: `float` (1,018).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 419 | 41.2% |
| `3` | 351 | 34.5% |
| `4` | 151 | 14.8% |
| `1` | 97 | 9.5% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: Self-isolation behavior ordinal scale; appears only in main survey

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Ordinal scale measuring degree of self-isolation behavior during COVID-19 pandemic, ranging from strict isolation to no behavior change. Includes detailed prompt explanation clarifying the definition of self-isolation.
