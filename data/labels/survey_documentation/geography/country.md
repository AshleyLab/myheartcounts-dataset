# country

**Benchmark column**: `field_country`
**Raw identifier**: `country`
**Role**: context
**Type**: categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~91
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> What is your country of residence?

## Answer options

| Value | Label |
|-------|-------|
| UK | United Kingdom |
| US | United States |
| HK | Hong Kong |

## Observed values

**Total observations**: 9,065 — **type-enforced**: 9,065 (**unique**: 3) — raw Python types seen: `str` (9,065).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `0` (US) | 7,906 | 87.2% |
| `1` (UK) | 1,032 | 11.4% |
| `2` (HK) | 127 | 1.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Data constraints
- **Data type**: string
- **Constraint type**: MultiValueConstraints
- **Allow multiple**: false
- **Allow other**: false

## Conditional logic
Country selection determines subsequent questions via skip rules:
- **UK** → skip to `zip` (UK postcode entry)
- **US** → skip to `numericZip` (US numeric zipcode entry)
- **HK** → skip to `riskfactors1` (skip zipcode, proceed to risk factors)
- **Other** → skip to `riskfactors1` (default, skip zipcode)

## Git history (file-level)
- Recent change: `5ff65f1` (2020-04-03) [MHC-756] Update for postal code
- Commits affecting file: 5
- Notes: Country classification with conditional routing. Updated in MHC-756 for postal code handling. Initial implementation 2015.

## Notes
- Context variable for geographic stratification and regional analysis
- Drives conditional flow in survey (determines if/how zipcode is collected)
- Three major geographic regions: UK (primary), US (primary), HK (secondary)
- Used for regional health comparisons and understanding geographic variation in cardiovascular health
- Determines data collection approach for postal information
