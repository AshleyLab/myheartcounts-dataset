# alcohol

**Benchmark column**: `field_alcohol`
**Raw identifier**: `alcohol`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~113-148
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How often do you consume alcoholic beverages?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Never |
| 2 | Once a month or less |
| 3 | 2-4 times a month |
| 4 | 2-3 times per week |
| 5 | 4 times or more per week |

**Data type**: integer  
**Input**: list (radio, allowMultiple: false)

## Observed values

**Total observations**: 956 — **type-enforced**: 956 (**unique**: 5) — raw Python types seen: `float` (956).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 227 | 23.7% |
| `2` | 227 | 23.7% |
| `3` | 205 | 21.4% |
| `4` | 165 | 17.3% |
| `5` | 132 | 13.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: 496824f 2020-04-03 [MHC-756] Add alcohol question to the diet survey
- Notes: Alcohol question added in commit 496824f (2020-04-03). Formatted in subsequent commit a244471 same day.

## Notes
5-point ordinal alcohol-consumption frequency. Source parquet stores the value as a single-element list (`list<double>`); the build extractor uses the `list_unwrap` transform to unbox. Earlier doc revisions called this column "fohol" (a benchmark naming quirk), but the actual emitted column is `field_alcohol` (corrected 2026-04-27).
