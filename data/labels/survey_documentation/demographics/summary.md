# Demographics

User-identifying traits: age, sex/gender, ethnicity/race, education, and innate physiological characteristics (skin type). Variables here are collected across multiple sources (Heart Age form, CVhealth survey, HealthKit characteristics) but are grouped together because they describe *who the participant is*. Geographic variables (country, zip) live in `geography/`.

## Variables (7 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [age](age.md) | target | continuous | Heart Age form | Age entered by user for risk calculation |
| [BiologicalSex](BiologicalSex.md) | target | binary | Heart Age form | Male / Female (Framingham stratification) |
| [ethnicity](ethnicity.md) | context | ordinal | cardio_CVhealth_survey.json | Hispanic/Latino origin (5 options) |
| [Ethnicity_heartage](Ethnicity_heartage.md) | context | categorical | Heart Age form | African-American vs Other (for Framingham) |
| [race](race.md) | context | categorical | cardio_CVhealth_survey.json | Self-identified race (12 options, multi-select) |
| [education](education.md) | context | ordinal | cardio_CVhealth_survey.json | Highest education level (7 options) |
| [FitzpatrickSkinType](FitzpatrickSkinType.md) | context | categorical → ordinal (released bucketed) | HealthKit (`HKCharacteristicTypeIdentifierFitzpatrickSkinType`) | Fitzpatrick skin type (raw I-VI categorical; released as bucketed `{light, medium, dark}` ordinal). Benchmark column: `field_FitzpatrickSkinType`. Earlier revisions misspelled this as `FrickSkinType`. |

## Notes

- `ethnicity` (Hispanic/Latino origin) and `race` (multi-select categories) are distinct questions in the CVhealth survey.
- `Ethnicity_heartage` is a *separate* 2-category ethnicity field used by the Framingham model — not the same as `ethnicity` above.
