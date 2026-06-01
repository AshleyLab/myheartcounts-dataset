# BMI_categories

**Benchmark column**: `BMI_categories`
**Raw identifier**: Derived from `BMI_values` (not directly surveyed)
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of BMI_values using WHO standard cutoff thresholds
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `BMI_values` (calculated as weight_kg / height_m^2)

## Question
Not directly asked — derived from height and weight values collected via the app, which are then used to compute BMI, which is then binned into categories.

## Derivation details

BMI (Body Mass Index) is calculated as weight in kilograms divided by height in meters squared. The resulting continuous BMI value is then classified into ordinal categories using WHO standard cutoffs:

- **Underweight**: BMI < 18.5
- **Normal weight**: 18.5 ≤ BMI < 25.0
- **Overweight**: 25.0 ≤ BMI < 30.0
- **Obese**: BMI ≥ 30.0

The exact binning cutpoints are applied in the MHC-benchmark repo's post-processing step (see benchmark repo for the precise implementation and any additional category refinements, e.g., obesity subclasses).

## Observed values

**Total observations**: 22,334 — **type-enforced**: 22,334 (**unique**: 5) — raw Python types seen: `str` (22,334).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (Normal weight) | 7,642 | 34.2% |
| `2` (Overweight) | 7,638 | 34.2% |
| `3` (Obesity) | 4,739 | 21.2% |
| `0` (Underweight) | 1,465 | 6.6% |
| `4` (Morbid Obesity) | 850 | 3.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related input: `BMI_values` (documented separately)
- Binning applied in MHC-benchmark post-processing (specific commits in MHC-benchmark repo, not shown here)

## Notes
- BMI_categories is **ordinal**: categories have a natural order (underweight < normal < overweight < obese).
- This is a standard public health classification and does not reflect individual-level risk assessment; see clinical guidelines for interpretation.
- Cross-reference: see `BMI_values.md` for the continuous input variable.
- The post-hoc nature means this variable is calculated after data extraction from the iOS app, in the analysis pipeline.
