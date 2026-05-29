# unhealthy

**Benchmark column**: `field_unhealthy`
**Raw identifier**: `unhealthy`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json`
- Line: ~16-62
- Survey: `Adequacy_of_activity_mindset_measure` (Adequacy of Activity Mindset)

## Question
> My current level of physical activity is unhealthy.

## Answer options
| Value | Label |
|-------|-------|
| 1 | Strongly Disagree |
| 2 | Disagree |
| 3 | Somewhat Disagree |
| 4 | Neither Agree or Disagree |
| 5 | Somewhat Agree |
| 6 | Agree |
| 7 | Strongly Agree |

## Observed values

**Total observations**: 1,131 — **type-enforced**: 1,131 (**unique**: 7) — raw Python types seen: `float` (1,131).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `5` | 242 | 21.4% |
| `2` | 233 | 20.6% |
| `1` | 197 | 17.4% |
| `3` | 146 | 12.9% |
| `6` | 124 | 11.0% |
| `4` | 113 | 10.0% |
| `7` | 76 | 6.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `ca99760` MHC-610 Add Adequacy of Activity Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Adequacy of Activity Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates agreement with the statement that one's current physical activity level is unhealthy on a 7-point Likert scale (strongly disagree to strongly agree).
