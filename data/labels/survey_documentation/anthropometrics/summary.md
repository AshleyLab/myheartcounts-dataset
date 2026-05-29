# Anthropometrics

Body-size measures collected via HealthKit (weight, height, BMI) and one derived ordinal target that bins BMI into WHO weight categories.

## Variables (4 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [WeightKilograms](WeightKilograms.md) | target | continuous | HealthKit `HKQuantityTypeIdentifierBodyMass` | Body weight in kg |
| [HeightCentimeters](HeightCentimeters.md) | context | continuous | HealthKit `HKQuantityTypeIdentifierHeight` | Height (read in meters, converted to cm) |
| [BMI_values](BMI_values.md) | target | continuous | HealthKit `HKQuantityTypeIdentifierBodyMassIndex` or computed from weight/height | Body Mass Index |
| [BMI_categories](BMI_categories.md) | target | ordinal | Derived (post-hoc binning in MHC-benchmark) | Underweight / normal / overweight / obese |

## Notes

- BMI typically computed at analysis time from HealthKit BodyMass + Height; the iOS app registers both quantity types in `APHAppDelegate.m:1346-1363`.
- `BMI_categories` binning cutpoints (WHO standard) are defined in the MHC-benchmark repo, not in this iOS codebase.
