# blood_pressure_categories

**Benchmark column**: `blood_pressure_categories`
**Raw identifier**: Derived from `SystolicBloodPressure` and `DiastolicBloodPressure`
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of systolic and diastolic blood pressure values using AHA (American Heart Association) guidelines
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `SystolicBloodPressure` (mmHg), `DiastolicBloodPressure` (mmHg)

## Question
Not directly asked — derived from blood pressure measurements entered into the app during the Heart Age survey.

## Derivation details

Blood pressure is classified into ordinal categories using AHA/ACC guidelines based on systolic and diastolic thresholds. A reading is assigned the most severe category applicable:

- **Normal**: Systolic < 120 AND Diastolic < 80
- **Elevated**: Systolic 120–129 AND Diastolic < 80
- **Stage 1 Hypertension**: Systolic 130–139 OR Diastolic 80–89
- **Stage 2 Hypertension**: Systolic ≥ 140 OR Diastolic ≥ 90

The binning logic applies the AHA/ACC 2017 guidelines. Exact implementation details (handling of boundary cases, whether both must be satisfied or either, priority order) are in the MHC-benchmark post-processing step.

## Observed values

**Total observations**: 9,867 — **type-enforced**: 9,867 (**unique**: 4) — raw Python types seen: `str` (9,867).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` (Hypertension_Stage_1) | 3,895 | 39.5% |
| `0` (Normal) | 3,373 | 34.2% |
| `1` (Elevated) | 1,673 | 17.0% |
| `3` (Hypertension_Stage_2) | 926 | 9.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related inputs: `SystolicBloodPressure` and `DiastolicBloodPressure` (documented separately)
- Binning applied in MHC-benchmark post-processing (specific commits in MHC-benchmark repo)

## Notes
- **Ordinal type**: categories have a natural order (normal < elevated < stage 1 < stage 2).
- Reflects AHA/ACC 2017 hypertension classification, not the older 140/90 threshold.
- This is post-hoc binning; the iOS app does not compute this variable directly.
- Cross-reference: see `SystolicBloodPressure.md` and `DiastolicBloodPressure.md` (if available) for raw input variables.
