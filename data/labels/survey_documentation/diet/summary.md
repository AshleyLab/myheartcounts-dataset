# Diet

Dietary intake from the Diet survey: five numeric food/drink frequency questions plus sodium-reduction behaviours (multi-select) and alcohol (5-point ordinal). All are context variables.

## Variables (7 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [fruit](fruit.md) | context | continuous | cardio_diet_survey.json | Cups of fruit per day |
| [vegetable](vegetable.md) | context | continuous | cardio_diet_survey.json | Cups of vegetables per day |
| [fish](fish.md) | context | continuous | cardio_diet_survey.json | Servings of fish per week |
| [grains](grains.md) | context | continuous | cardio_diet_survey.json | Servings of whole grains per day |
| [sugar_drinks](sugar_drinks.md) | context | continuous | cardio_diet_survey.json | Sugar-sweetened beverages per week |
| [sodium](sodium.md) | context | ordinal | cardio_diet_survey.json | Sodium-reduction behaviours (multi-select, 3 strategies) |
| [alcohol](alcohol.md) | context | ordinal | cardio_diet_survey.json | Alcohol consumption frequency (5-point ordinal). Benchmark column: `field_alcohol`. Earlier revisions used the misspelled name `fohol`. |

## Notes

- The benchmark column is `field_alcohol`; the source parquet stores the value as a single-element `list<double>` and the build extractor unboxes it via the `list_unwrap` transform. Earlier doc revisions called this column "fohol".
- There is no separate "eating mindset" / "reasons for eating" survey in the iOS app — the variables sometimes referred to as an "eating-reasons battery" in the benchmark actually live in `mindset_measures/` and are about *activity*, not eating.
