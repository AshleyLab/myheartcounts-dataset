# Tobacco, Vaping, and Cannabis

The full tobacco-and-vaping survey: current use, past use, quit attempts, onset ages, quit readiness, and multi-select product lists. Organized by substance (vaping, cigarettes, smokeless tobacco, cannabis). All items are context variables.

## Variables (28 files)

### Vaping (6)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [currentVaping](currentVaping.md) | context | ordinal | Current vaping frequency |
| [pastVaping](pastVaping.md) | context | ordinal | Ever vaped in the past (yes/no ordinal) |
| [onsetVaping](onsetVaping.md) | context | continuous | Age of first vape |
| [everQuitVaping](everQuitVaping.md) | context | binary | Tried to quit vaping in past 12 months |
| [durationQuitVaping](durationQuitVaping.md) | context | categorical | How long since last vaped (codes 1-3 ordered Days/Months/Years; 4=Never, 5=Don't know break ordering) |
| [readinessQuitVaping](readinessQuitVaping.md) | context | continuous | Readiness to quit vaping (1–10) |

### Cigarette smoking (5)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [currentSmoking](currentSmoking.md) | context | ordinal | Current cigarette frequency |
| [onsetSmoking](onsetSmoking.md) | context | continuous | Age of first cigarette |
| [everQuitSmoking](everQuitSmoking.md) | context | binary | Tried to quit in past 12 months |
| [durationQuitSmoking](durationQuitSmoking.md) | context | categorical | Duration since last cigarette (same encoding as durationQuitVaping) |
| [readinessQuitSmoking](readinessQuitSmoking.md) | context | continuous | Readiness to quit smoking (1–10) |

### Smokeless tobacco (6)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [currentSmokeless](currentSmokeless.md) | context | ordinal | Current smokeless use frequency |
| [pastSmokeless](pastSmokeless.md) | context | ordinal | Ever used smokeless in past |
| [onsetSmokeless](onsetSmokeless.md) | context | continuous | Age of first smokeless use |
| [everQuitSmokeless](everQuitSmokeless.md) | context | binary | Tried to quit smokeless in past 12 months |
| [durationQuitSmokeless](durationQuitSmokeless.md) | context | categorical | Duration since last smokeless use (same encoding as durationQuitVaping) |
| [readinessQuitSmokeless](readinessQuitSmokeless.md) | context | continuous | Readiness to quit smokeless (1–10) |

### Cannabis (4)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [cannabisSmoking](cannabisSmoking.md) | context | ordinal | Cannabis smoking history |
| [currentCannabisSmoking](currentCannabisSmoking.md) | context | ordinal | Current cannabis smoking frequency |
| [pastCannabisSmoking](pastCannabisSmoking.md) | context | ordinal | Duration of past cannabis use |
| [lastCannabisSmoking](lastCannabisSmoking.md) | context | ordinal | Time since last cannabis use |

### Product multi-selects (2)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [tobaccoProducts](tobaccoProducts.md) | context | multi_categorical | Tobacco products used in past week (multi-select; 8 codes) |
| [tobaccoProductsEver](tobaccoProductsEver.md) | context | multi_categorical | Tobacco products ever used (multi-select; 8 codes) |

## Notes

- Source: `cardio_vaping_and_smoking_survey.json`.
- No cannabis-onset-age variable exists in the iOS survey. The benchmark's `cannabis_onset_age` context (if used) is not sourced from this iOS repo.
- Smoking is a Framingham input — see `framingham_risk.md` in `cardiometabolic_labs/`.
- Branching logic: quit-related items are typically only asked when the respondent reports past or current use.
