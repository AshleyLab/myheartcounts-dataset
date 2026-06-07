# GoSleepTime

**Benchmark column**: `field_GoSleepTime`
**Raw identifier**: `GoSleepTime`
**Role**: context
**Type**: continuous

## Source
- **NOT in survey JSON**: `GoSleepTime` does not appear in `cardio_activitysleep_survey.json` or any other survey in `CardioHealth/Resources/JSONs/cardiosurveys/`.
- **AppCore Profile Item**: Managed as `kAPCUserInfoItemTypeSleepTime` in AppCore framework (note: AppCore uses `SleepTime` internally; mapped to `GoSleepTime` in benchmark column).
- File reference: `CardioHealth/Startup/APHAppDelegate.m` line ~1436
- See also: `APCUser+Sleep.h` (from AppCore)

## Question
Not a survey question. This is a user profile preference.

> What time do you usually go to sleep? (user profile setting)

## Answer options
Continuous time value (hours and minutes), typically 0–24 hour format or user's local time.

| Constraint | Value |
|-----------|-------|
| Data Type | time |
| Source | User profile / HealthKit sleep analysis |
| UI Hint | time picker (profile settings) |

## Observed values

**Total observations**: 25,349 — **type-enforced**: 25,349 (**unique**: 533) — raw Python types seen: `float` (25,349).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 15.00 |
| median | 19.50 |
| mean | 17.25 |
| q75 | 21.50 |
| max | 24.00 |
| std | 6.42 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `21.50` | 2,485 |
| `20.50` | 2,143 |
| `19.50` | 1,965 |
| `22.50` | 1,852 |
| `23.50` | 1,188 |
| `18.50` | 1,176 |
| `21.00` | 947 |
| `20.00` | 927 |
| `15.50` | 870 |
| `22.00` | 846 |
| `14.50` | 776 |
| `23.00` | 773 |
| `0.50` | 619 |
| `0` | 530 |
| `13.50` | 493 |
| `19.00` | 476 |
| `15.00` | 469 |
| `16.50` | 429 |
| `16.00` | 363 |
| `1.50` | 308 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Derived from AppCore framework integration, not specific to `cardio_activitysleep_survey.json`.
- Not tracked in Activity/Sleep survey file history.
- See AppCore repository for profile management history.

## Notes
- **Not extracted from survey**: Unlike `sleep_time` and `sleep_time1` which are explicit survey questions, `GoSleepTime` is a user-configurable profile preference.
- **HealthKit integration**: May also be derived or validated against HealthKit sleep analysis (see `HKObject+Metadata` in AppDelegate imports).
- **Defines sleep window**: Together with `WakeUpTime`, establishes the user's expected sleep period for context-aware coaching and activity tracking.
- **Coaching contextualization**: Used to avoid sending activity/stand reminders during expected sleep hours.
- **Complement to WakeUpTime**: Provides boundaries for wake-time activity windows and sleep-time coaching (e.g., sleep hygiene advice).

## Related variables
- `WakeUpTime`: Complement to `GoSleepTime`; together they define active/sleep window.
- `sleep_time`, `sleep_time1`: Survey-based sleep duration measures (see `cardio_activitysleep_survey.json`).
- `sleep_diagnosis1`, `sleep_diagnosis2`: Clinical sleep disorder classification (see `cardio_activitysleep_survey.json`).
