# activity1_type

**Benchmark column**: `field_activity1_type`
**Raw identifier**: `activity1_type`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~74 (daily_check) and ~82 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> Which activity did you do that may not have been recorded by your phone or wearable?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Walking |
| 2 | Jogging |
| 3 | Cycling |
| 4 | Tennis or other racquet sport |
| 5 | Soccer, basketball, or other team sport |
| 6 | Weight-lifting |
| 7 | Swimming |
| (other) | Custom text entry |

## Observed values

**Total observations**: 15,062 — **type-enforced**: 15,062 (**unique**: 7) — raw Python types seen: `str` (15,062).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `0` (walking) | 6,887 | 45.7% |
| `5` (weight_lifting) | 3,395 | 22.5% |
| `1` (jogging) | 1,603 | 10.6% |
| `2` (cycling) | 1,371 | 9.1% |
| `6` (swimming) | 849 | 5.6% |
| `4` (team_sports) | 663 | 4.4% |
| `3` (tennis) | 294 | 2.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable activity list since initial commit; allows single selection with `allowOther: true`

## Notes
This variable specifies the type of first unrecorded activity, shown only if `activity1_option` is true. It is a categorical variable with 7 predefined options and an open-ended "other" option. See also: `activity1_option`, `activity1_time`, `activity1_intensity`, and the second-activity counterparts `activity2_type`.
