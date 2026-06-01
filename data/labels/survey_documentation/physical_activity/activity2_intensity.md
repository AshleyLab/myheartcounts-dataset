# activity2_intensity

**Benchmark column**: `field_activity2_intensity`
**Raw identifier**: `activity2_intensity`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~258 (daily_check) and ~266 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> How intense was the activity?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Light |
| 2 | Moderate |
| 3 | Vigorous |

## Observed values

**Total observations**: 5,972 — **type-enforced**: 5,972 (**unique**: 3) — raw Python types seen: `str` (5,972).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (moderate) | 2,954 | 49.5% |
| `0` (light) | 2,036 | 34.1% |
| `2` (vigorous) | 982 | 16.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable intensity scale since initial commit; single-select question

## Notes
This variable records the intensity level of the second unrecorded activity, shown only if `activity2_option` is true. It is an ordinal variable ranging from light to vigorous, allowing stratification of activity benefit. See also: `activity2_option`, `activity2_type`, `activity2_time`.
