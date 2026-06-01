# pleasurable

**Benchmark column**: `field_pleasurable`
**Raw identifier**: `pleasurable`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json`
- Line: ~51-82
- Survey: `Exercise_process_mindset_measure` (Exercise Process Mindset)

## Question
> EXERCISING is:

## Answer options
| Value | Label |
|-------|-------|
| 1 | Very unpleasant |
| 2 | Somewhat unpleasant |
| 3 | Somewhat pleasurable |
| 4 | Very pleasurable |

## Observed values

**Total observations**: 1,161 — **type-enforced**: 1,161 (**unique**: 4) — raw Python types seen: `float` (1,161).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 555 | 47.8% |
| `2` | 308 | 26.5% |
| `4` | 221 | 19.0% |
| `1` | 77 | 6.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `f608626` MHC-610 Add Exercise Process Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Exercise Process Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates users' perception of pleasure/enjoyment in exercising on a 4-point scale (very unpleasant to very pleasurable).
