# framingham_risk

**Benchmark column**: `framingham_risk`
**Raw identifier**: (derived — no direct survey field)
**Role**: target
**Type**: continuous

## Source
- Derivation: Framingham Risk Equation for Hard ASCVD (Atherosclerotic Cardiovascular Disease) 10-year risk
- iOS calculation: APHHeartAgeAndRiskFactors.m, lines 173-271
- Input variables: Age, Gender, Ethnicity, TotalCholesterol, HDL, SystolicBloodPressure, Hypertension (treatment status), Smoking, Diabetes

## Question
Not directly asked — derived from the Heart Age survey. Users answer demographic and clinical questions that populate the risk factor inputs.

## Derivation details

The Framingham 10-year risk for hard ASCVD is computed using gender- and ethnicity-specific coefficients. The formula follows a logistic risk model with a baseline 10-year survival rate and population mean adjusted for log-transformed risk factors.

**Core formula**:
```
individualEstimatedTenYearRisk = 1 - baseline^exp(individualSum - populationMean)
```

**individualSum calculation** (sum of 13 weighted terms):
1. `log(Age) * coef[0]`
2. `log(Age)^2 * coef[1]`
3. `log(TotalCholesterol) * coef[2]`
4. `log(Age) * log(TotalCholesterol) * coef[3]`
5. `log(HDL-C) * coef[4]`
6. `log(Age) * log(HDL-C) * coef[5]`
7. `log(SystolicBP_treated) * coef[6]` (where SystolicBP_treated = log(SystolicBP) if Hypertension==1, else 0)
8. `log(Age) * log(SystolicBP_treated) * coef[7]`
9. `log(SystolicBP_untreated) * coef[8]` (where SystolicBP_untreated = log(SystolicBP) if Hypertension==0, else 0)
10. `log(Age) * log(SystolicBP_untreated) * coef[9]`
11. `Smoking * coef[10]`
12. `log(Age) * Smoking * coef[11]`
13. `Diabetes * coef[12]`

**Coefficients** vary by gender and ethnicity (stratified in lookup table):
- Female African-American, Female Other (non-African-American)
- Male African-American, Male Other (non-African-American)

Each gender-ethnicity stratum has:
- 13 coefficients (indexed 0–12 above)
- A baseline 10-year survival probability
- A population mean for risk adjustment

**Output**: A continuous probability between 0 and 1 (or 0–100 if scaled to percentage), representing the estimated 10-year risk of hard ASCVD.

## Observed values

**Total observations**: 5,968 — **type-enforced**: 5,968 (**unique**: 5,718) — raw Python types seen: `float` (5,968).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.003377 |
| q25 | 0.03 |
| median | 0.06 |
| mean | 0.08 |
| q75 | 0.10 |
| max | 0.57 |
| std | 0.07 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `0.01` | 7 |
| `0.02` | 4 |
| `0.04` | 4 |
| `0.04` | 4 |
| `0.06` | 4 |
| `0.09` | 4 |
| `0.01` | 3 |
| `0.02` | 3 |
| `0.02` | 3 |
| `0.02` | 3 |
| `0.02` | 3 |
| `0.02` | 3 |
| `0.03` | 3 |
| `0.03` | 3 |
| `0.03` | 3 |
| `0.04` | 3 |
| `0.04` | 3 |
| `0.04` | 3 |
| `0.05` | 3 |
| `0.05` | 3 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source file or calculation)
- **APHHeartAgeAndRiskFactors.m**: Recent material change
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support (added Pulmonary Hypertension to survey)
  - `c312938` MHC-626 Upgrade to ResearchKit 2.0
  - `06a6f76` MHX-640 Added NSLocalizedString for unlocalized strings

## Notes
- This is the core 10-year CVD risk variable computed in the iOS app during the Heart Age survey.
- Ethnicity is remapped: "Black" → "African-American"; all others → "Other".
- Hypertension status (0 or 1) is used to split the systolic BP term into treated/untreated contributions.
- If the calculation yields NaN (not a number), it is replaced with 0.
- The corresponding "Heart Age" is derived from this risk by reverse-lookup into a precomputed table (see `findHeartAgeForRiskValue:forGender:forEthnicity:` method, lines 316–347).
