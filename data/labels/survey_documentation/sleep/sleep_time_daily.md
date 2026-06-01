# sleep_time_daily

**Benchmark column**: `field_sleep_time_daily`
**Raw identifier**: `sleep_time` (daily check variant)
**Role**: context
**Type**: continuous (duration in minutes)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~301 (daily_check) and ~296 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> How many hours of sleep did you get last night?

**Detail**: Use the below scroll to indicate how long, in hours and minutes, you think you slept last night.

## Answer options
| Value | Label |
|-------|-------|
| duration | Numeric input (minutes), via timepicker UI with 15-minute steps |

## Observed values

**Total observations**: 34,617 — **type-enforced**: 34,617 (**unique**: 543) — raw Python types seen: `float` (34,617).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 60.00 |
| q25 | 2.154e+04 |
| median | 2.52e+04 |
| mean | 2.23e+04 |
| q75 | 2.88e+04 |
| max | 8.634e+04 |
| std | 9824 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 4,089 |
| `2.52e+04` | 3,674 |
| `2.88e+04` | 3,402 |
| `2.16e+04` | 2,428 |
| `2.526e+04` | 2,188 |
| `2.886e+04` | 1,999 |
| `2.166e+04` | 1,484 |
| `2.7e+04` | 1,198 |
| `3.24e+04` | 1,127 |
| `2.34e+04` | 1,099 |
| `1.8e+04` | 942 |
| `1.806e+04` | 661 |
| `3.246e+04` | 612 |
| `3.06e+04` | 527 |
| `1.98e+04` | 464 |
| `3.6e+04` | 462 |
| `1.44e+04` | 367 |
| `1.446e+04` | 254 |
| `2.61e+04` | 254 |
| `3.606e+04` | 216 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable question since initial commit; data type is `duration`, unit is `minutes`, step is 15

## Notes
This variable captures sleep duration for the prior night, collected daily via the daily check-in survey. It is distinct from other sleep-related variables (e.g., `sleep_time` from other surveys, sleep diagnosis questions, sleep category variables from other questionnaires). This daily sleep_time is a continuous context variable that may correlate with activity levels and provide longitudinal sleep tracking. The 15-minute granularity supports both hours and minutes entry via the timepicker UI.
