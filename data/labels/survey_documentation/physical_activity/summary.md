# Physical Activity

Self-reported physical activity across leisure-time and occupational contexts, plus fine-grained daily check-in items logging the first and second activity sessions of the day (type, duration, intensity). Spans the activity/sleep survey and the daily-check survey.

## Variables (13 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [vigorous_act](vigorous_act.md) | target | continuous | cardio_activitysleep_survey.json | Minutes of vigorous activity per week |
| [moderate_act](moderate_act.md) | context | continuous | cardio_activitysleep_survey.json | Minutes of moderate activity per week |
| [work](work.md) | target | binary | cardio_activitysleep_survey.json | Has regular work (gates `atwork`) |
| [atwork](atwork.md) | context | ordinal | cardio_activitysleep_survey.json | Work-time physical intensity (4-point) |
| [phys_activity](phys_activity.md) | context | ordinal | cardio_activitysleep_survey.json | Leisure-time activity level (6-point) |
| [activity1_option](activity1_option.md) | context | binary | cardio_daily_check.json | First-activity-of-day option flag |
| [activity2_option](activity2_option.md) | context | binary | cardio_daily_check.json | Second-activity-of-day option flag |
| [activity1_type](activity1_type.md) | context | categorical | cardio_daily_check.json | Activity 1 type (walking/jogging/cycling/…) |
| [activity2_type](activity2_type.md) | context | categorical | cardio_daily_check.json | Activity 2 type |
| [activity1_time](activity1_time.md) | context | continuous | cardio_daily_check.json | Activity 1 duration (minutes) |
| [activity2_time](activity2_time.md) | context | continuous | cardio_daily_check.json | Activity 2 duration (minutes) |
| [activity1_intensity](activity1_intensity.md) | context | ordinal | cardio_daily_check.json | Activity 1 intensity (light/moderate/vigorous) |
| [activity2_intensity](activity2_intensity.md) | context | ordinal | cardio_daily_check.json | Activity 2 intensity |

## Notes

- `vigorous_act` and `moderate_act` are weekly retrospective self-report.
- `activity1_*` / `activity2_*` are *daily* check-in items logged on the day they happen.
- `atwork` is only shown when `work == true`.
- Wearable-derived activity (steps, energy burned) lives in `healthkit_watch_metrics/`, not here.
