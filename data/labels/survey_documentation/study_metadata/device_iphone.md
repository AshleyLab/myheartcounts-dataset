# device_iphone

**Benchmark column**: `field_device_iphone`
**Raw identifier**: `device` (multi-select option with value `1`)
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
| `True` | 40,045 | 89.0% |
| `False` | 4,968 | 11.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 4
- Recent material change: `a24fdfe` (MHC-355 - Updates to "Get started" (Day One) survey)
- Notes: Multi-select question with `allowMultiple: true`. This document corresponds to the iPhone option (value = 1).

## Notes
This is one option of the `device` multi-select question in day_one.json. The benchmark variable `field_device_iphone` represents whether the iPhone option was selected (true/1 if selected, false/0 if not). See also: `device_smartwatch`, `device_activity_band`, `device_other`.
