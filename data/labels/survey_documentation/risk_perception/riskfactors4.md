# riskfactors4

**Benchmark column**: `field_riskfactors4`
**Raw identifier**: `riskfactors4`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~313
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Over your lifetime, compared to others your age and sex, how would you rate your risk of having a heart attack, stroke, or dying due to cardiovascular disease? (choose one)

## Answer options

| Value | Label |
|-------|-------|
| 1 | Much lower than average |
| 2 | Lower than average |
| 3 | Average |
| 4 | Higher than average |
| 5 | Much higher than average |

## Observed values

**Total observations**: 30,487 — **type-enforced**: 30,487 (**unique**: 5) — raw Python types seen: `float` (30,487).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 8,860 | 29.1% |
| `2` | 8,681 | 28.5% |
| `4` | 6,102 | 20.0% |
| `1` | 5,877 | 19.3% |
| `5` | 967 | 3.2% |

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
- Measures comparative perception of lifetime cardiovascular risk relative to age and sex peers
- Context variable for understanding health beliefs and long-term relative risk perception
- Part of the cardiovascular risk perception assessment module
- Related to riskfactors3 (absolute lifetime risk) but measures comparative rather than absolute risk
- Similar response scale to riskfactors2 but with lifetime horizon instead of 10-year
- Used to assess alignment between objective lifetime risk and perceived relative risk
