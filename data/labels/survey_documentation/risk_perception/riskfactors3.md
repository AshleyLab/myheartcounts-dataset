# riskfactors3

**Benchmark column**: `field_riskfactors3`
**Raw identifier**: `riskfactors3`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~273
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Over your lifetime how likely do you think it is that you personally will have a heart attack, stroke, or die due to cardiovascular disease? (choose one)

## Answer options

| Value | Label |
|-------|-------|
| 1 | Not at all |
| 2 | A little |
| 3 | Moderately |
| 4 | A lot |
| 5 | Extremely |

## Observed values

**Total observations**: 30,511 — **type-enforced**: 30,511 (**unique**: 5) — raw Python types seen: `float` (30,511).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 11,940 | 39.1% |
| `3` | 8,816 | 28.9% |
| `1` | 5,039 | 16.5% |
| `4` | 3,412 | 11.2% |
| `5` | 1,304 | 4.3% |

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
- Measures subjective perception of lifetime cardiovascular risk
- Context variable for understanding health beliefs and long-term risk perception
- Part of the cardiovascular risk perception assessment module
- Related to riskfactors1 (10-year risk) but measures lifetime rather than 10-year risk horizon
- Similar response scale to riskfactors1, facilitating comparison between time horizons
- Used to assess alignment between objective lifetime risk and perceived risk
