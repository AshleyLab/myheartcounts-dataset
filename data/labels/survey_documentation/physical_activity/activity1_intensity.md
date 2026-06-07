# activity1_intensity

**Benchmark column**: `field_activity1_intensity`
**Raw identifier**: `activity1_intensity`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~139 (daily_check) and ~147 (daily_check_coaching)
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

**Total observations**: 18,442 — **type-enforced**: 18,442 (**unique**: 3) — raw Python types seen: `str` (18,442).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (moderate) | 8,985 | 48.7% |
| `0` (light) | 5,845 | 31.7% |
| `2` (vigorous) | 3,612 | 19.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable intensity scale since initial commit; single-select question

## Notes
This variable records the intensity level of the first unrecorded activity, shown only if `activity1_option` is true. It is an ordinal variable ranging from light to vigorous, allowing stratification of activity benefit. See also: `activity1_option`, `activity1_type`, `activity1_time`.
