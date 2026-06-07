# Cardiovascular Disease History

Self-reported diagnoses of heart and vascular disease, family history, current cardiovascular medications, and a family of derived binary flags for specific subtypes (Afib, CAD, heart failure, pulmonary hypertension, etc.). The two source multi-select questions — `heart_disease` and `vascular` — feed all the subtype flags.

## Variables (12 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [heart_disease](heart_disease.md) | context | categorical (multi-select) | cardio_CVhealth_survey.json | 11 heart-disease diagnosis options |
| [vascular](vascular.md) | context | categorical (multi-select) | cardio_CVhealth_survey.json | 8 vascular-disease diagnosis options |
| [family_history](family_history.md) | context | categorical (multi-select) | cardio_CVhealth_survey.json | Family history of early heart disease |
| [medications_to_treat](medications_to_treat.md) | context | categorical (multi-select) | cardio_CVhealth_survey.json | CV/metabolic medications in use |
| [cardiovascular_disease](cardiovascular_disease.md) | target | binary | Derived | Any positive answer across heart_disease or vascular |
| [CAD](CAD.md) | target | binary | Derived | Coronary artery disease (MI / coronary blockage / angina) |
| [<Heart Failure or CHF.md>](<Heart Failure or CHF.md>) | target | binary | Derived | heart_disease option value 7 |
| [<Atrial fibrillation (Afib).md>](<Atrial fibrillation (Afib).md>) | target | binary | Derived | heart_disease option value 8 |
| [PH](PH.md) | target | binary | Derived | Pulmonary hypertension (heart_disease option 11; added in MHC-734) |
| [<Congenital Heart.md>](<Congenital Heart.md>) | target | binary | Derived | heart_disease option value 9 |
| [<Peripheral-Systemic Vascular Disease.md>](<Peripheral-Systemic Vascular Disease.md>) | target | binary | Derived | vascular option (peripheral vascular disease) |
| [<Cerebrovascular Disease.md>](<Cerebrovascular Disease.md>) | target | binary | Derived | vascular options (stroke / TIA) |

## Notes

- All derived binary targets one-hot the multi-select enumeration values from `heart_disease` (in heart_disease.md) or `vascular` (in vascular.md). See those two source files for the authoritative option lists.
- `PH` support was added in commit MHC-734; earlier data will not have this flag populated.
- `PAH` (Pulmonary Arterial Hypertension) was added as a separate `vascular` option in MHC-780 but is not in the benchmark target list.
