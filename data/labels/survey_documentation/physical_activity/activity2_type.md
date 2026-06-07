# activity2_type

**Benchmark column**: `field_activity2_type`
**Raw identifier**: `activity2_type`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~193 (daily_check) and ~201 (daily_check_coaching)
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

**Total observations**: 4,464 — **type-enforced**: 4,464 (**unique**: 7) — raw Python types seen: `str` (4,464).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `0` (walking) | 2,088 | 46.8% |
| `5` (weight_lifting) | 1,067 | 23.9% |
| `1` (jogging) | 497 | 11.1% |
| `2` (cycling) | 399 | 8.9% |
| `6` (swimming) | 204 | 4.6% |
| `4` (team_sports) | 147 | 3.3% |
| `3` (tennis) | 62 | 1.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable activity list since initial commit; allows single selection with `allowOther: true`

## Notes
This variable specifies the type of second unrecorded activity, shown only if `activity2_option` is true. It is a categorical variable with the same 7 predefined options and open-ended "other" option as `activity1_type`. See also: `activity2_option`, `activity2_time`, `activity2_intensity`, and the first-activity counterparts `activity1_type`.
