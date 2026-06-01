# riskfactors2

**Benchmark column**: `field_riskfactors2`
**Raw identifier**: `riskfactors2`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~233
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Over the next 10 years, compared to others your age and sex, how would you rate your risk of having a heart attack, stroke, or dying due to cardiovascular disease? (choose one)

## Answer options

| Value | Label |
|-------|-------|
| 1 | Much lower than average |
| 2 | Lower than average |
| 3 | Average |
| 4 | Higher than average |
| 5 | Much higher than average |

## Observed values

**Total observations**: 30,529 — **type-enforced**: 30,529 (**unique**: 5) — raw Python types seen: `float` (30,529).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 8,883 | 29.1% |
| `3` | 8,162 | 26.7% |
| `1` | 7,319 | 24.0% |
| `4` | 5,491 | 18.0% |
| `5` | 674 | 2.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Data constraints
- **Data type**: integer
- **Enumeration type**: MultiValueConstraints
- **Allow multiple**: false
- **Allow other**: false

## Git history (file-level)
- Recent change: `5ff65f1` (2020-04-03) [MHC-756] Update for postal code
- Commits affecting file: 5
- Notes: Cardiovascular risk perception question. Stable since initial implementation; recent changes unrelated to this question.

## Notes
- Measures comparative perception of 10-year cardiovascular risk relative to age and sex peers
- Context variable for understanding health beliefs and risk perception
- Part of the cardiovascular risk perception assessment module
- Related to riskfactors1 (absolute risk) but measures comparative rather than absolute risk
- Used to assess alignment between objective risk factors and perceived relative risk
