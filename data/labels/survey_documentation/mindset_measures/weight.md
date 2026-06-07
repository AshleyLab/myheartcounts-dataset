# weight

**Benchmark column**: `field_weight`
**Raw identifier**: `weight`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json`
- Line: ~66-112
- Survey: `Adequacy_of_activity_mindset_measure` (Adequacy of Activity Mindset)

## Question
> My current level of physical activity is helping me achieve or maintain a healthy body weight.

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

**Total observations**: 1,128 — **type-enforced**: 1,128 (**unique**: 7) — raw Python types seen: `float` (1,128).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `6` | 239 | 21.2% |
| `5` | 216 | 19.1% |
| `3` | 169 | 15.0% |
| `2` | 148 | 13.1% |
| `4` | 138 | 12.2% |
| `7` | 128 | 11.3% |
| `1` | 90 | 8.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `ca99760` MHC-610 Add Adequacy of Activity Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Adequacy of Activity Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates agreement with the statement that one's current physical activity level is helping achieve or maintain a healthy body weight on a 7-point Likert scale (strongly disagree to strongly agree).
