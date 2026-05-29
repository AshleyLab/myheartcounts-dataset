# WakeUpTime

**Benchmark column**: `field_WakeUpTime`
**Raw identifier**: `WakeUpTime`
**Role**: context
**Type**: continuous

## Source
- **NOT in survey JSON**: `WakeUpTime` does not appear in `cardio_activitysleep_survey.json` or any other survey in `CardioHealth/Resources/JSONs/cardiosurveys/`.
- **AppCore Profile Item**: Managed as `kAPCUserInfoItemTypeWakeUpTime` in AppCore framework.
- File reference: `CardioHealth/Startup/APHAppDelegate.m` line ~1435
- See also: `APCUser+Sleep.h` (from AppCore)

## Question
Not a survey question. This is a user profile preference.

> What time do you usually wake up? (user profile setting)

## Answer options
Continuous time value (hours and minutes), typically 0–24 hour format or user's local time.

| Constraint | Value |
|-----------|-------|
| Data Type | time |
| Source | User profile / HealthKit sleep analysis |
| UI Hint | time picker (profile settings) |

## Observed values

**Total observations**: 25,262 — **type-enforced**: 25,262 (**unique**: 597) — raw Python types seen: `float` (25,262).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 4.00 |
| median | 6.00 |
| mean | 8.22 |
| q75 | 8.25 |
| max | 23.97 |
| std | 6.66 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `4.00` | 2,634 |
| `5.00` | 2,490 |
| `7.00` | 2,184 |
| `6.00` | 1,841 |
| `23.00` | 1,266 |
| `3.00` | 1,132 |
| `8.00` | 1,095 |
| `4.50` | 691 |
| `0` | 643 |
| `3.50` | 619 |
| `5.50` | 611 |
| `22.00` | 543 |
| `6.50` | 510 |
| `2.00` | 500 |
| `9.00` | 427 |
| `7.50` | 358 |
| `2.50` | 315 |
| `1.00` | 309 |
| `16.00` | 309 |
| `23.50` | 291 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Derived from AppCore framework integration, not specific to `cardio_activitysleep_survey.json`.
- Not tracked in Activity/Sleep survey file history.
- See AppCore repository for profile management history.

## Notes
- **Not extracted from survey**: Unlike `sleep_time` and `sleep_time1` which are explicit survey questions, `WakeUpTime` is a user-configurable profile preference.
- **HealthKit integration**: May also be derived or validated against HealthKit sleep analysis (see `HKObject+Metadata` in AppDelegate imports).
- **Coaching trigger**: Used in `APHStandModule.m` (`isWakeUpTime` method) to determine if user is currently in their wake period for stand/activity coaching interventions.
- **Companion to GoSleepTime**: Together with `GoSleepTime`, defines the user's active window for coaching and real-time activity monitoring.
- **Contextual**: Helps tailor activity suggestions and stand reminders to times when user is awake and likely to respond.

## Related variables
- `GoSleepTime`: Complement to `WakeUpTime`; together they define active/sleep window.
- `sleep_time`, `sleep_time1`: Survey-based sleep duration measures (see `cardio_activitysleep_survey.json`).
- `sleep_diagnosis1`, `sleep_diagnosis2`: Clinical sleep disorder classification (see `cardio_activitysleep_survey.json`).
