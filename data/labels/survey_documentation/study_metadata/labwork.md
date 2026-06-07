# labwork

**Benchmark column**: `field_labwork`
**Raw identifier**: `labwork`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json`
- Line: ~69
- Survey: `day_one`

## Question
> After 7 days, you will be asked to calculate your heart risk score, which requires entering your cholesterol blood values and blood pressure. Will you have those data in the next 7 days?

## Answer options
| Value | Label |
|-------|-------|
| 1 | I currently have this information |
| 2 | I will try to get this information during the next week |

## Observed values

**Total observations**: 41,322 — **type-enforced**: 41,322 (**unique**: 2) — raw Python types seen: `float` (41,322).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` (2) | 31,344 | 75.9% |
| `1` (1) | 9,978 | 24.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 4
- Recent material change: `a24fdfe` (MHC-355 - Updates to "Get started" (Day One) survey)
- Notes: Stable question since initial commit; last modified in MHC-355 update to Day One survey

## Notes
This variable captures whether the participant already has or can obtain lab work (cholesterol and blood pressure values) during the study's first week, which is required for the heart risk score calculation (Framingham or AHA score).
