# Sleep

Sleep duration, bedtime/wake-time, sleep disorder diagnosis, and derived ordinal categories. Spans the activity/sleep survey, the daily check-in, and AppCore user-profile items for bedtime/wake-time.

## Variables (10 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [sleep_time](sleep_time.md) | context | continuous | cardio_activitysleep_survey.json | Hours of sleep needed to feel rested |
| [sleep_time1](sleep_time1.md) | context | continuous | cardio_activitysleep_survey.json | Actual weekday hours slept |
| [sleep_time_daily](sleep_time_daily.md) | context | continuous | cardio_daily_check.json | Minutes slept last night (daily) |
| [sleep_time_categories](sleep_time_categories.md) | target | ordinal | Derived | Ordinal bins of nightly sleep duration |
| [WakeUpTime](WakeUpTime.md) | context | continuous | AppCore profile (`kAPCUserInfoItemTypeWakeUpTime`) | Typical wake-up time |
| [WakeUpTime_categories](WakeUpTime_categories.md) | target | ordinal | Derived | Ordinal bins of wake-up time |
| [GoSleepTime](GoSleepTime.md) | context | continuous | AppCore profile (`kAPCUserInfoItemTypeSleepTime`) | Typical bedtime |
| [GoSleepTime_categories](GoSleepTime_categories.md) | target | ordinal | Derived | Ordinal bins of bedtime |
| [sleep_diagnosis1](sleep_diagnosis1.md) | target | binary | cardio_activitysleep_survey.json | Has diagnosed sleep disorder (gates sleep_diagnosis2) |
| [sleep_diagnosis2](sleep_diagnosis2.md) | context | categorical | cardio_activitysleep_survey.json | Type of sleep disorder (8-option multi-select) |

## Notes

- Three distinct sleep-duration variables: survey-retrospective (`sleep_time`), weekday (`sleep_time1`), and daily check-in (`sleep_time_daily`).
- `WakeUpTime` and `GoSleepTime` are *not* survey questions — they are AppCore user-info profile items set during onboarding (see `APHAppDelegate.m:1435-1436`).
- The `_categories` variables are post-hoc binnings defined in the MHC-benchmark repo.
