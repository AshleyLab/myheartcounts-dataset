# activity1_option

**Benchmark column**: `field_activity1_option`
**Raw identifier**: `activity1_option`
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~51 (daily_check) and ~59 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> Did you perform any physical activities yesterday that you think were not recorded by your phone or wearable device?

**Detail**: Some activities, such as weight lifting, may not be fully recorded by activity sensors.

## Answer options
| Value | Label |
|-------|-------|
| 0 | No (skips to sleep_time) |
| 1 | Yes |

## Observed values

**Total observations**: 34,627 — **type-enforced**: 34,627 (**unique**: 2) — raw Python types seen: `bool` (34,627).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 25,870 | 74.7% |
| `True` | 8,757 | 25.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable skip-logic question present since initial commit; skip rule diverts to `sleep_time` if answered "No"

## Notes
This is a checkpoint question that branches the survey: if the respondent reports unrecorded activities, they are asked to specify activity details (type, duration, intensity); if not, the survey skips to sleep_time. This variable captures whether the respondent self-reports sensor-undetected activity. See also: `activity2_option`, `activity1_type`, `activity1_time`, `activity1_intensity`.
