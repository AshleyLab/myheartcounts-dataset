# Cardiometabolic Labs

Lab values and clinical measurements entered through the Heart Age form (cholesterol panel, blood pressure, glucose, diabetes/hypertension status) plus the derived Framingham 10-year ASCVD risk that the app computes from them. Blood-pressure categorisation is also derived here.

## Variables (10 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [SystolicBloodPressure](SystolicBloodPressure.md) | target | continuous | Heart Age form | Systolic BP (mmHg) |
| [DiastolicBloodPressure](DiastolicBloodPressure.md) | context | continuous | Heart Age form | Diastolic BP (mmHg) |
| [blood_pressure_categories](blood_pressure_categories.md) | target | ordinal | Derived | AHA categories (normal / elevated / stage-1 / stage-2) |
| [TotalCholesterol](TotalCholesterol.md) | target | continuous | Heart Age form | Total cholesterol (mg/dL) |
| [Hdl](Hdl.md) | target | continuous | Heart Age form | HDL cholesterol |
| [Ldl](Ldl.md) | target | continuous | Heart Age form | LDL cholesterol (collected but not used in Framingham) |
| [BloodGlucose](BloodGlucose.md) | context | continuous | Heart Age form | Blood glucose |
| [Diabetes](Diabetes.md) | target | binary | Heart Age form | Diabetes diagnosis (Y/N) |
| [Hypertension](Hypertension.md) | target | binary | Heart Age form | On anti-hypertensive treatment (Y/N) |
| [framingham_risk](framingham_risk.md) | target | continuous | Computed in `APHHeartAgeAndRiskFactors.m` | 10-year hard ASCVD risk (Framingham) |

## Notes

- `framingham_risk` inputs: age, sex (`BiologicalSex`), `Ethnicity_heartage`, `TotalCholesterol`, `Hdl`, `SystolicBloodPressure`, `Hypertension` (treatment status), smoking, `Diabetes`. Age/sex/ethnicity live in `demographics/`; smoking lives in `tobacco_vaping_cannabis/`.
- `Hypertension` records treatment status, not measured high BP — a critical distinction for Framingham.
- `Ldl` and `BloodGlucose` are collected for research but do not feed the Framingham calculation.
