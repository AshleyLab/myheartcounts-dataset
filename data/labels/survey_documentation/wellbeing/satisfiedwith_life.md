# satisfiedwith_life

**Benchmark column**: `satisfiedwith_life`
**Raw identifier**: `satisfiedwith_life`
**Role**: target
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~9
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Overall, how satisfied are you with life as a whole these days?

## Answer options
Scale from 0 to 10 where:
- 0 = "not at all satisfied"
- 10 = "completely satisfied"

| Value | Label |
|-------|-------|
| 0 | Not at all satisfied |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | Completely satisfied |

## Observed values

**Total observations**: 30,563 — **type-enforced**: 30,387 (**unique**: 4) — raw Python types seen: `str` (30,387), `float` (176).
**Type-enforcement rejections**: 176 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (High) | 14,348 | 47.2% |
| `0` (Very High) | 7,534 | 24.8% |
| `2` (Medium) | 5,428 | 17.9% |
| `3` (Low) | 3,077 | 10.1% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `High` | 14,348 |
| `Very High` | 7,534 |
| `Medium` | 5,428 |
| `Low` | 3,077 |

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
- Notes: Part of ONS wellbeing framework. This variable has been stable since initial implementation (2015); most recent change was postal code update unrelated to this question.

## Notes
- This is a core ONS (Office for National Statistics) wellbeing framework question
- Forms the primary satisfaction measure in the wellbeing assessment
- Associated with broader psychological wellbeing tracking in the study
