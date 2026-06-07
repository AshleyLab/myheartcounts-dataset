# phone_on_user

**Benchmark column**: `field_phone_on_user`
**Raw identifier**: `phone_on_user`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~12 (daily_check) and ~24 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> In the last 24 hours, how often did you have your phone or wearable device with you?

**Detail**: Keeping your phone or device on you will help to better understand your daily activity, which will help the study.

## Answer options

The benchmark stores string codes (not numeric); `ordinal_dictionary.json::field_phone_on_user` provides the int ordering used by the API:

| String code | API int | Label |
|---|---:|---|
| `all_the_time` | 0 | All the time |
| `all_day_and_night` | 1 | All day and night (24/7) |
| `all_day_not_night` | 2 | All day but not at night |
| `most_of_the_time` | 3 | Most of the time |
| `half_of_time` | 4 | About half of the time |
| `rarely_if_at_all` | 5 | Rarely if at all |

The earlier 4-option list in this doc (matching the older `cardio_daily_check.json` schema) was incomplete; the merged daily-check tables emit 6 distinct strings (verified in `data/labels/context_labels.json` 2026-04-27).

## Observed values

**Total observations**: 34,648 — **type-enforced**: 34,648 (**unique**: 6) — raw Python types seen: `str` (34,648).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` (all_day_not_night) | 12,220 | 35.3% |
| `1` (all_day_and_night) | 11,277 | 32.5% |
| `4` (half_of_time) | 4,525 | 13.1% |
| `0` (all_the_time) | 2,712 | 7.8% |
| `3` (most_of_the_time) | 2,640 | 7.6% |
| `5` (rarely_if_at_all) | 1,274 | 3.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable question present since initial commit to both daily check variants

## Notes
This variable measures compliance with device-wearing, an important context variable that affects the quality of passive activity tracking data. It appears in both the standard daily check-in and the coaching variant daily check-in. Respondents with "Rarely if at all" (value 4) should be interpreted as having potentially incomplete activity sensor data.
