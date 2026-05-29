# riskfactors1

**Benchmark column**: `field_riskfactors1`
**Raw identifier**: `riskfactors1`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~193
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Over the next 10 years how likely do you think it is that you personally will have a heart attack, stroke, or die due to cardiovascular disease? (choose one)

## Answer options

| Value | Label |
|-------|-------|
| 1 | Not at all |
| 2 | A little |
| 3 | Moderately |
| 4 | A lot |
| 5 | Extremely |

## Observed values

**Total observations**: 30,558 — **type-enforced**: 30,558 (**unique**: 5) — raw Python types seen: `float` (30,558).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 13,018 | 42.6% |
| `2` | 11,340 | 37.1% |
| `3` | 4,633 | 15.2% |
| `4` | 1,184 | 3.9% |
| `5` | 383 | 1.3% |

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
- Measures subjective perception of 10-year cardiovascular risk
- Context variable for understanding health beliefs and risk perception
- Part of the cardiovascular risk perception assessment module
- Related to riskfactors2 (comparative risk) but measures absolute rather than relative risk
- Used to assess alignment between objective risk factors and perceived risk
