# Gender

> **Not in released benchmark.** Reason: redundancy — duplicate of target `BiologicalSex`. See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `Gender`
**Raw identifier**: `heartAgeDataGender`
**Obj-C constant**: `kHeartAgeTestDataGender`
**Role**: context
**Type**: categorical

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 74 (also defined: line 50–51 for gender value constants)
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 177, 187, 263, 276, 316, 363, 408
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

**Notes on values**: Stored as string values `Female` or `Male` (constants `kHeartAgeTestDataGenderFemale` and `kHeartAgeTestDataGenderMale` at lines 75–76). "Other" is presented as an option but is not stratified in the Framingham lookup tables; it defaults to "Other" category.

## Observed values

**Total observations**: 3,769 — **type-enforced**: 0 (**unique**: 0) — raw Python types seen: `str` (3,769).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 3,769 dictionary-miss (`KeyError`).

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `"Male"` | 2,822 |
| `"Female"` | 919 |
| `` | 19 |
| `"Other"` | 9 |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `06a6f76`, `2a31f49` (MHC-327)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- This is the **same variable as `BiologicalSex`** above. Both the benchmark column `Gender` (context) and `BiologicalSex` (target) refer to the same raw identifier `heartAgeDataGender`.
- Gender is **contextual** to the Framingham calculation in that it determines which set of coefficients are used (separate tables exist for females and males). The gender-specific coefficients are keyed by the string values `Female` and `Male` in the lookup table (lines 105, 136).
- Can be pre-populated from HealthKit biological sex when user selects "Are you submitting your own heart risk data?" = YES.
- The Framingham coefficients show notably different age and cholesterol effects by gender, reflecting epidemiological differences in cardiovascular risk between sexes.
