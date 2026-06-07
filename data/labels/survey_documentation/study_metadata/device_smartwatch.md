# device_smartwatch

**Benchmark column**: `field_device_smartwatch`
**Raw identifier**: `device` (multi-select option with value `3`)
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json`
- Line: ~36
- Survey: `day_one`

## Question
> Which devices do you use to track activity and sleep?

**Note**: Please plan to have these on you during the study period.

## Answer options
| Value | Label |
|-------|-------|
| 1 | iPhone |
| 2 | Activity band or pedometer |
| 3 | Smartwatch / Apple Watch |

## Observed values

**Total observations**: 45,013 — **type-enforced**: 45,013 (**unique**: 2) — raw Python types seen: `bool` (45,013).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 33,710 | 74.9% |
| `True` | 11,303 | 25.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 4
- Recent material change: `a24fdfe` (MHC-355 - Updates to "Get started" (Day One) survey)
- Notes: Multi-select question with `allowMultiple: true`. This document corresponds to the Smartwatch / Apple Watch option (value = 3).

## Notes
This is one option of the `device` multi-select question in day_one.json. The benchmark variable `field_device_smartwatch` represents whether the smartwatch option was selected (true/1 if selected, false/0 if not). See also: `device_iphone`, `device_activity_band`, `device_other`.
