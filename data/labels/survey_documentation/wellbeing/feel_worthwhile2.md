# feel_worthwhile2

**Benchmark column**: `feel_worthwhile2`
**Raw identifier**: `feel_worthwhile2`
**Role**: target
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~41
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> How about happy?

## Answer options
Scale from 0 to 10 measuring yesterday's experience where:
- 0 = "did not experience the feeling at all yesterday"
- 10 = "experienced the feeling all of the time yesterday"

| Value | Label |
|-------|-------|
| 0 | Not at all |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | All of the time |

## Observed values

**Total observations**: 30,553 — **type-enforced**: 30,383 (**unique**: 4) — raw Python types seen: `str` (30,383), `float` (170).
**Type-enforcement rejections**: 170 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (High) | 13,152 | 43.3% |
| `0` (Very High) | 8,060 | 26.5% |
| `2` (Medium) | 5,735 | 18.9% |
| `3` (Low) | 3,436 | 11.3% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `High` | 13,152 |
| `Very High` | 8,060 |
| `Medium` | 5,735 |
| `Low` | 3,436 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Data constraints
- **Data type**: integer
- **Min value**: 0
- **Max value**: 10
- **Step**: 1
- **UI hint**: slider

## Git history (file-level)
- Recent change: `5ff65f1` (2020-04-03) [MHC-756] Update for postal code
- Commits affecting file: 5
- Notes: Part of ONS daily experience questions. Stable since initial implementation.

## Notes
- This is a core ONS (Office for National Statistics) wellbeing framework question measuring daily emotional experience
- Specifically assesses happiness/positive affect experienced the previous day
- Part of a sequence of daily feeling questions (feel_worthwhile2-4)
- Related to (but distinct from) the broader life satisfaction and worthwhile questions (satisfiedwith_life and feel_worthwhile1)
- Similar to the daily `happiness` variable but operates within the wellbeing survey context
