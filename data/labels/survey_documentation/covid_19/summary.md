# COVID-19

Items from the one-time COVID-19 survey and its recurrent (bi-weekly) counterpart. Captures test results, symptom burden, healthcare utilisation, exposure, protective behaviours, and household impact. All context variables.

## Variables (16 files)

### Test & diagnosis (2)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [covid](covid.md) | context | ordinal | RNA test result status (4-level) |
| [covid_serologic](covid_serologic.md) | context | ordinal | Antibody test status (3-level) |

### Symptoms (2 multi-selects + 2 severity sliders)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [symptoms_week_preceding](symptoms_week_preceding.md) | context | multi_categorical | Symptoms in week preceding test (multi-select; 12 codes) |
| [symptoms_past_week](symptoms_past_week.md) | context | multi_categorical | Symptoms in past week (multi-select; 12 codes; recurrent: `symptoms_past_2_weeks`) |
| [severity_covid](severity_covid.md) | context | ordinal | Worst COVID-19 severity (0–10 slider) |
| [severity](severity.md) | context | ordinal | General health severity (0–10 slider) |

### Healthcare utilisation (3)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [most_intense_care](most_intense_care.md) | context | ordinal | Highest care level received (5-level) |
| [daily_activities](daily_activities.md) | context | ordinal | Functional capacity during illness (4-level) |
| [days_admitted](days_admitted.md) | context | continuous | Days hospitalised |

### Household & exposure (3)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [exposure](exposure.md) | context | ordinal | COVID-19 exposure intensity (3-level) |
| [building](building.md) | context | ordinal | People living in participant's building (4-level density) |
| [healthcare_worker](healthcare_worker.md) | context | categorical | Healthcare worker position type |

### Protective behaviours (2)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [self_isolating](self_isolating.md) | context | ordinal | Self-isolation behaviour (4-level) |
| [face_covering](face_covering.md) | context | ordinal | Face covering compliance (4-level) |

### Conditions & medications (2)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [conditions](conditions.md) | context | multi_categorical | Pre-existing conditions (multi-select; 9 codes) |
| [antibiotics](antibiotics.md) | context | multi_categorical | Current medications/treatments (multi-select; 12 codes) |

## Notes

- Sources: `cardio_covid_19_survey.json` (one-time) and `cardio_covid_19_recurrent_survey.json` (bi-weekly).
