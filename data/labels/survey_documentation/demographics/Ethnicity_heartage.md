# Ethnicity_heartage

**Benchmark column**: `field_Ethnicity_heartage`
**Raw identifier**: `heartAgeDataEthnicity`
**Obj-C constant**: `kHeartAgeTestDataEthnicity`
**Role**: context
**Type**: categorical

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 48
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 178, 180–184, 187, 263, 277, 316, 363
- UI question: `APHHeartAgeTaskViewController.m` lines 195-210
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Ethnicity

## Answer options
| Value | Label |
|-------|-------|
| I prefer not to indicate an ethnicity | I prefer not to indicate an ethnicity |
| Alaska Native | Alaska Native |
| American Indian | American Indian |
| Asian | Asian |
| Black | Black |
| Hispanic | Hispanic |
| Pacific Islander | Pacific Islander |
| White | White |
| Other | Other |

**Notes on values**: Nine localized string options are presented to the user. However, only two categories are used internally for the Framingham calculation: "African-American" (when user selects "Black") and "Other" (all other selections, including prefer-not-to-answer).

## Observed values

**Total observations**: 10,210 — **type-enforced**: 10,210 (**unique**: 9) — raw Python types seen: `str` (10,210).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `6` (White) | 7,891 | 77.3% |
| `2` (Asian) | 862 | 8.4% |
| `4` (Hispanic) | 688 | 6.7% |
| `3` (Black) | 320 | 3.1% |
| `7` (Other) | 253 | 2.5% |
| `8` (I prefer not to indicate an ethnicity) | 115 | 1.1% |
| `1` (American Indian) | 46 | 0.5% |
| `5` (Pacific Islander) | 30 | 0.3% |
| `0` (Alaska Native) | 5 | 0.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update), `06a6f76`
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Ethnicity is **contextual** to the Framingham calculation; it determines which set of coefficients are used.
- The user-visible options (9 categories) are mapped internally to only 2 Framingham categories (lines 180–184):
  - "Black" → `kLookupEthnicityAfricanAmerican` ("African-American")
  - All others → `kLookupEthnicityOther` ("Other")
- The Framingham model stratifies by African-American vs. Other race/ethnicity, reflecting different baseline risks and coefficient values documented in epidemiological literature.
- Appears as a single-choice question on its own form step, marked as required.
- The "Black" category is the only one explicitly recognized by the Framingham lookup; all other selections collapse into the "Other" category.
