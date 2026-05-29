# activity1_time

**Benchmark column**: `field_activity1_time`
**Raw identifier**: `activity1_time`
**Role**: context
**Type**: continuous (duration in minutes)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json` and `cardio_daily_check_coaching.json`
- Line: ~123 (daily_check) and ~132 (daily_check_coaching)
- Survey: `daily_check` (both standard and coaching variants)

## Question
> How long did you do the activity?

## Answer options
| Value | Label |
|-------|-------|
| duration | Numeric input (minutes), via timepicker UI |

## Observed values

**Total observations**: 18,484 — **type-enforced**: 18,484 (**unique**: 313) — raw Python types seen: `float` (18,484).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 60.00 |
| q25 | 900 |
| median | 2400 |
| mean | 3889 |
| q75 | 3660 |
| max | 8.634e+04 |
| std | 5850 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 3,206 |
| `3600` | 2,411 |
| `1800` | 1,980 |
| `1200` | 1,016 |
| `3660` | 856 |
| `7200` | 818 |
| `900` | 730 |
| `2700` | 687 |
| `600` | 576 |
| `5400` | 558 |
| `7260` | 425 |
| `2400` | 388 |
| `1.08e+04` | 298 |
| `1500` | 275 |
| `2100` | 247 |
| `4500` | 199 |
| `1.086e+04` | 195 |
| `300` | 177 |
| `1.44e+04` | 176 |
| `3000` | 171 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3 (daily_check.json), 4 (daily_check_coaching.json)
- Recent material change: `7f52783` (MHC-626 - Fix parsing survey element without createdOn property)
- Notes: Stable duration question since initial commit; data type is `duration`, unit is `minutes`, no step constraint

## Notes
This variable records the duration of the first unrecorded activity, shown only if `activity1_option` is true. The answer is collected via a timepicker UI and stored as minutes. This allows continuous quantification of self-reported unrecorded activity. See also: `activity1_option`, `activity1_type`, `activity1_intensity`.
