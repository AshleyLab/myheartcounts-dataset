# muscles

**Benchmark column**: `field_muscles`
**Raw identifier**: `muscles`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json`
- Line: ~216-262
- Survey: `Adequacy_of_activity_mindset_measure` (Adequacy of Activity Mindset)

## Question
> How much is your current level of physical (in-)activity strengthening or weakening your muscles?

## Answer options
| Value | Label |
|-------|-------|
| 7 | Strengthening very much |
| 6 | Strengthening moderately |
| 5 | Strengthening slightly |
| 4 | Neither strengthening nor weakening |
| 3 | Weakening slightly |
| 2 | Weakening moderately |
| 1 | Weakening very much |

## Observed values

**Total observations**: 1,127 — **type-enforced**: 1,127 (**unique**: 7) — raw Python types seen: `float` (1,127).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `5` | 256 | 22.7% |
| `6` | 248 | 22.0% |
| `4` | 218 | 19.3% |
| `3` | 149 | 13.2% |
| `7` | 111 | 9.8% |
| `2` | 90 | 8.0% |
| `1` | 55 | 4.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Most recent commit: `7f52783` MHC-626 - Fix parsing survey element without createdOn property (2025-02-14)
- Initial addition: `ca99760` MHC-610 Add Adequacy of Activity Mindset Measure Survey (2025-02-13)
- Notes: Survey was added as part of MHC-610, then had a parsing fix in MHC-626

## Notes
- Item from the Adequacy of Activity Mindset battery.
- The user's benchmark list refers to these as "eating-reasons battery (13)" — this appears to be a misnomer. The actual surveys are about physical activity/exercise mindsets, not eating. The `cardio_exercise_process_mindset_measure_survey.json` and `cardio_adequacy_of_activity_mindset_measure_survey.json` together comprise the activity mindset items.
- This item evaluates perceived muscle strengthening/weakening impact of one's current physical activity level. Note: values are reverse-coded compared to other items (7=strongest benefit, 1=strongest negative).
