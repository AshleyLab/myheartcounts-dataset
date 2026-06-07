# BiologicalSex

**Benchmark column**: `BiologicalSex`
**Raw identifier**: `heartAgeDataGender`
**Obj-C constant**: `kHeartAgeTestDataGender`
**Role**: target
**Type**: binary

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 74
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 177, 187–189
- UI question: `APHHeartAgeTaskViewController.m` lines 165-178
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Gender

## Answer options
| Value | Label |
|-------|-------|
| `Female` | Female |
| `Male` | Male |
| `Other` | Other |

**Notes**: Mapped from HKBiologicalSex enum at display time (lines 167-169). The stored value is the string value of the HKBiologicalSex enum constant (`HKBiologicalSexFemale`, `HKBiologicalSexMale`, `HKBiologicalSexOther`), which is converted to the lookup value (`Female` or `Male`) at processing time (lines 575-585).

## Observed values

**Total observations**: 25,217 — **type-enforced**: 25,217 (**unique**: 2) — raw Python types seen: `str` (25,217).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `True` | 18,614 | 73.8% |
| `False` | 6,603 | 26.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0`, `06a6f76`, `2a31f49` (MHC-327)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Gender is critical to the Framingham calculation as separate coefficient tables are maintained per gender and ethnicity combination (males have higher risk in the Framingham model).
- Can be pre-populated from HealthKit when user selects "Are you submitting your own heart risk data?" = YES (line 686-690).
- The lookup uses only `Female` and `Male` for the Framingham coefficients; any selection of "Other" defaults to one of the two stratified models.
