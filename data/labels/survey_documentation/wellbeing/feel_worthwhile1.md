# feel_worthwhile1

**Benchmark column**: `feel_worthwhile1`
**Raw identifier**: `feel_worthwhile1`
**Role**: target
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~25
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Overall, to what extent do you feel the things you do in your life are worthwhile?

## Answer options
Scale from 0 to 10 where:
- 0 = "not at all worthwhile"
- 10 = "completely worthwhile"

| Value | Label |
|-------|-------|
| 0 | Not at all worthwhile |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | Completely worthwhile |

## Observed values

**Total observations**: 30,545 — **type-enforced**: 30,403 (**unique**: 4) — raw Python types seen: `str` (30,403), `float` (142).
**Type-enforcement rejections**: 142 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (High) | 12,649 | 41.6% |
| `0` (Very High) | 9,669 | 31.8% |
| `2` (Medium) | 5,148 | 16.9% |
| `3` (Low) | 2,937 | 9.7% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `High` | 12,649 |
| `Very High` | 9,669 |
| `Medium` | 5,148 |
| `Low` | 2,937 |

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
- Notes: Part of ONS wellbeing framework. Stable since initial implementation; recent changes were unrelated to this question.

## Notes
- This is a core ONS (Office for National Statistics) wellbeing framework question
- Measures sense of purpose and meaning in life activities
- Related but distinct from general life satisfaction (satisfiedwith_life)
- Companion to daily experience questions (feel_worthwhile2-4)
