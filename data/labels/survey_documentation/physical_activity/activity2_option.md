# activity2_option

**Benchmark column**: `field_activity2_option`
**Raw identifier**: `activity2_option`
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~170 (daily_check) and ~178 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> Did you perform any additional physical activities yesterday that you think were not recorded by your phone or wearable device?

## Answer options
| Value | Label |
|-------|-------|
| 0 | No (skips to sleep_time) |
| 1 | Yes |

## Observed values

**Total observations**: 18,499 — **type-enforced**: 18,499 (**unique**: 2) — raw Python types seen: `bool` (18,499).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 15,261 | 82.5% |
| `True` | 3,238 | 17.5% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable skip-logic question present since initial commit; skip rule diverts to `sleep_time` if answered "No"

## Notes
This is the second activity branching question, following the same pattern as `activity1_option` but for additional unrecorded activities. If the respondent reports a second unrecorded activity, they specify details (type, duration, intensity); if not, the survey skips to sleep_time. See also: `activity1_option`, `activity2_type`, `activity2_time`, `activity2_intensity`.
