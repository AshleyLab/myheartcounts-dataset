# Study Metadata

Onboarding and participation-context variables: device ownership, lab-work availability, and daily device-wearing compliance. Not about health directly but about the participant's ability to contribute data.

## Variables (6 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [labwork](labwork.md) | context | binary | cardio_day_one.json | Will have lab work in next 7 days |
| [device_iphone](device_iphone.md) | context | binary | cardio_day_one.json (device multi-select) | Owns iPhone |
| [device_smartwatch](device_smartwatch.md) | context | binary | cardio_day_one.json (device multi-select) | Owns Apple Watch / smartwatch |
| [device_activity_band](device_activity_band.md) | context | binary | cardio_day_one.json (device multi-select) | Owns activity band / pedometer |
| [device_other](device_other.md) | context | binary | cardio_day_one.json (device multi-select) | Owns other device |
| [phone_on_user](phone_on_user.md) | context | ordinal | cardio_daily_check.json | How often phone is on their person (4-level) |

## Notes

- The four `device_*` flags all derive from a single `device` multi-select element in `cardio_day_one.json`. Each flag is one option being checked.
- `phone_on_user` is a daily compliance measure — relevant for interpreting activity sensor data completeness.
