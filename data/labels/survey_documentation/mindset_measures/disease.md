# disease

**Benchmark column**: `field_disease`
**Raw identifier**: `disease`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json`
- Line: ~166-212
- Survey: `Adequacy_of_activity_mindset_measure` (Adequacy of Activity Mindset)

## Question
> How much does your current level of physical (in-)activity increase or decrease your risk of disease?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Increases my risk very much |
| 2 | Increases my risk moderately |
| 3 | Increases my risk slightly |
| 4 | Neither increases nor decreases my risk |
| 5 | Decreases my risk slightly |
| 6 | Decreases my risk moderately |
| 7 | Decreases my risk very much |

## Observed values

**Total observations**: 1,127 — **type-enforced**: 1,127 (**unique**: 7) — raw Python types seen: `float` (1,127).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `6` | 220 | 19.5% |
| `5` | 188 | 16.7% |
| `3` | 184 | 16.3% |
| `7` | 168 | 14.9% |
| `4` | 153 | 13.6% |
| `2` | 139 | 12.3% |
| `1` | 75 | 6.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `ca99760` MHC-610 Add Adequacy of Activity Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Adequacy of Activity Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates perceived disease risk impact of one's current physical activity level on a 7-point scale (increases risk very much to decreases risk very much).
