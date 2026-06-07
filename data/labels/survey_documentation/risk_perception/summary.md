# Risk Perception

Four self-rated CVD risk perception items from the Wellbeing survey. Each is a 5-point ordinal scale asking the participant to estimate their own risk of heart disease at different horizons and in absolute vs. relative terms.

## Variables (4 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [riskfactors1](riskfactors1.md) | context | ordinal | cardio_wellbeing_survey.json | 10-year absolute CVD risk (5-point) |
| [riskfactors2](riskfactors2.md) | context | ordinal | cardio_wellbeing_survey.json | 10-year risk vs others same age/sex (5-point) |
| [riskfactors3](riskfactors3.md) | context | ordinal | cardio_wellbeing_survey.json | Lifetime absolute CVD risk (5-point) |
| [riskfactors4](riskfactors4.md) | context | ordinal | cardio_wellbeing_survey.json | Lifetime risk vs others same age/sex (5-point) |

## Notes

- Separate from — and potentially divergent from — the computed `framingham_risk` in `cardiometabolic_labs/`. These capture *perceived* risk, not measured risk.
- Compare perceived vs. calculated risk for insight into participants' risk literacy.
