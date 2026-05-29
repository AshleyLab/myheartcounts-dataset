# device_other

**Benchmark column**: `field_device_other`
**Raw identifier**: `device` (multi-select with `allowOther: true`)
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
| (other) | Custom text entry |

## Observed values

**Total observations**: 45,013 — **type-enforced**: 45,013 (**unique**: 2) — raw Python types seen: `bool` (45,013).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 43,766 | 97.2% |
| `True` | 1,247 | 2.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 4
- Recent material change: `a24fdfe` (MHC-355 - Updates to "Get started" (Day One) survey)
- Notes: Multi-select question with `allowMultiple: true` and `allowOther: true`. This document corresponds to the custom "other" option.

## Notes
This is the "other" category of the `device` multi-select question in day_one.json. The benchmark variable `field_device_other` represents whether the participant entered a custom device not among the predefined options (true/1 if "other" was selected, false/0 if not). See also: `device_iphone`, `device_smartwatch`, `device_activity_band`.
