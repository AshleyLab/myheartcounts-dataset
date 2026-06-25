# Geography

Where the participant is located. Both variables are user-entered free-text in the Wellbeing survey and are anonymised before release (UK postcodes collapse to `"UK"`, Hong Kong tokens collapse to `"HK"`, US zips are kept as 1–3-digit truncations with HIPAA-listed low-population codes dropped).

## Variables (2 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [country](country.md) | context | categorical | cardio_wellbeing_survey.json | Country of residence (US / UK / HK) |
| [zip](zip.md) | context | categorical | cardio_wellbeing_survey.json | UK postcode / US zip (anonymised) |

## Notes

- `country` and `zip` previously lived in `demographics/`; they were split out into `geography/` so the demographics group cleanly isolates *who the participant is* from *where they are*.
- Both are released only after the anonymisation rules above; raw entries are not exposed.
