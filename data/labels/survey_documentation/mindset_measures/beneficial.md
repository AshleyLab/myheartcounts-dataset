# beneficial

**Benchmark column**: `field_beneficial`
**Raw identifier**: `beneficial`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json`
- Line: ~116-162
- Survey: `Adequacy_of_activity_mindset_measure` (Adequacy of Activity Mindset)

## Question
> How harmful/beneficial is your current level of physical activity for your health?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Very harmful for my health |
| 2 | Moderately harmful for my health |
| 3 | Slightly harmful for my health |
| 4 | Neither harmful nor beneficial for my health |
| 5 | Moderately beneficial for my health |
| 6 | Very beneficial for my health |
| 7 | Extremely beneficial for my health |

## Observed values

**Total observations**: 1,129 — **type-enforced**: 1,129 (**unique**: 7) — raw Python types seen: `float` (1,129).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `5` | 265 | 23.5% |
| `6` | 235 | 20.8% |
| `3` | 201 | 17.8% |
| `4` | 126 | 11.2% |
| `2` | 123 | 10.9% |
| `7` | 119 | 10.5% |
| `1` | 60 | 5.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `ca99760` MHC-610 Add Adequacy of Activity Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Adequacy of Activity Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates perceived harm/benefit of one's current physical activity level on a 7-point scale (very harmful to extremely beneficial).
