# activity2_time

**Benchmark column**: `field_activity2_time`
**Raw identifier**: `activity2_time`
**Role**: context
**Type**: continuous (duration in minutes)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~243 (daily_check) and ~251 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> How long did you do the activity?

## Answer options
| Value | Label |
|-------|-------|
| duration | Numeric input (minutes), via timepicker UI |

## Observed values

**Total observations**: 6,122 — **type-enforced**: 6,122 (**unique**: 195) — raw Python types seen: `float` (6,122).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 60.00 |
| q25 | 180 |
| median | 1800 |
| mean | 3093 |
| q75 | 3600 |
| max | 8.634e+04 |
| std | 5488 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 1,505 |
| `1800` | 667 |
| `3600` | 566 |
| `1200` | 412 |
| `900` | 329 |
| `3660` | 297 |
| `600` | 254 |
| `2700` | 184 |
| `7200` | 168 |
| `7260` | 155 |
| `5400` | 121 |
| `2400` | 109 |
| `1500` | 100 |
| `300` | 76 |
| `1.08e+04` | 73 |
| `1.086e+04` | 58 |
| `2100` | 51 |
| `4500` | 48 |
| `3000` | 39 |
| `1.44e+04` | 39 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable duration question since initial commit; data type is `duration`, unit is `minutes`, no step constraint

## Notes
This variable records the duration of the second unrecorded activity, shown only if `activity2_option` is true. The answer is collected via a timepicker UI and stored as minutes. This allows continuous quantification of self-reported unrecorded activity. See also: `activity2_option`, `activity2_type`, `activity2_intensity`.
