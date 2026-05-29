# feel_worthwhile3

**Benchmark column**: `feel_worthwhile3`
**Raw identifier**: `feel_worthwhile3`
**Role**: target
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~57
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> How about worried?

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

**Total observations**: 30,546 — **type-enforced**: 28,863 (**unique**: 4) — raw Python types seen: `str` (28,863), `float` (1,683).
**Type-enforcement rejections**: 1,683 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` (Low) | 13,905 | 48.2% |
| `2` (Medium) | 6,176 | 21.4% |
| `1` (High) | 6,164 | 21.4% |
| `0` (Very High) | 2,618 | 9.1% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `Low` | 13,905 |
| `Medium` | 6,176 |
| `High` | 6,164 |
| `Very High` | 2,618 |

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
- Specifically assesses worry/anxiety experienced the previous day
- Part of a sequence of daily feeling questions (feel_worthwhile2-4)
- Typically scored in reverse in wellbeing indices as higher worry indicates lower wellbeing
- Captures negative affect component of daily experience assessment
