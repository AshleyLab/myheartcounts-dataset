# Demographics

User-identifying traits: age, sex/gender, ethnicity/race, education, and innate physiological characteristics (blood type, skin type). Variables here are collected across multiple sources (Heart Age form, CVhealth survey, HealthKit characteristics, COVID survey) but are grouped together because they describe *who the participant is*. Geographic variables (country, zip) live in `geography/`.

## Variables (11 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [age](age.md) | target | continuous | Heart Age form | Age entered by user for risk calculation |
| [Age_heartage](Age_heartage.md) | context | continuous | Heart Age form | Same age field, duplicated for benchmark schema |
| [CurrentAge](CurrentAge.md) | context | continuous | Heart Age form | Current age pre-filled from HealthKit date of birth |
| [BiologicalSex](BiologicalSex.md) | target | binary | Heart Age form | Male / Female (Framingham stratification) |
| [Gender](Gender.md) | context | categorical | Heart Age form | Same as BiologicalSex (context-role duplicate) |
| [ethnicity](ethnicity.md) | context | ordinal | cardio_CVhealth_survey.json | Hispanic/Latino origin (5 options) |
| [Ethnicity_heartage](Ethnicity_heartage.md) | context | categorical | Heart Age form | African-American vs Other (for Framingham) |
| [race](race.md) | context | categorical | cardio_CVhealth_survey.json | Self-identified race (12 options, multi-select) |
| [education](education.md) | context | ordinal | cardio_CVhealth_survey.json | Highest education level (7 options) |
| [BloodType](BloodType.md) | context | categorical | cardio_covid_19_survey.json | ABO blood type (A/B/AB/O/Unknown) |
| [FitzpatrickSkinType](FitzpatrickSkinType.md) | context | categorical → ordinal (released bucketed) | HealthKit (`HKCharacteristicTypeIdentifierFitzpatrickSkinType`) | Fitzpatrick skin type (raw I-VI categorical; released as bucketed `{light, medium, dark}` ordinal). Benchmark column: `field_FitzpatrickSkinType`. Earlier revisions misspelled this as `FrickSkinType`. |

## Notes

- `age`, `Age_heartage`, and `CurrentAge` are three separate benchmark columns but all derive from the same Heart Age form input. `CurrentAge` is pre-filled from HealthKit date-of-birth; the other two record the value the user confirms.
- `BiologicalSex` (target) and `Gender` (context) share the underlying Heart Age constant `kHeartAgeTestDataGender`.
- `ethnicity` (Hispanic/Latino origin) and `race` (multi-select categories) are distinct questions in the CVhealth survey.
- `Ethnicity_heartage` is a *separate* 2-category ethnicity field used by the Framingham model — not the same as `ethnicity` above.
