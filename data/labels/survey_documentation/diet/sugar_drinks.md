# sugar_drinks

**Benchmark column**: `field_sugar_drinks`
**Raw identifier**: `sugar_drinks`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~64-74
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How many beverages with added sugar do you drink every week?

## Answer options
**Data type**: integer  
**Unit**: beverages  
**Range**: 0 to 50  
**Input**: numberfield

## Observed values

**Total observations**: 29,572 — **type-enforced**: 29,572 (**unique**: 42) — raw Python types seen: `float` (29,572).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 0 |
| median | 2.00 |
| mean | 3.88 |
| q75 | 5.00 |
| max | 50.00 |
| std | 6.14 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `0` | 10,329 |
| `1.00` | 3,751 |
| `2.00` | 3,371 |
| `3.00` | 2,182 |
| `5.00` | 2,171 |
| `7.00` | 1,731 |
| `10.00` | 1,263 |
| `4.00` | 1,198 |
| `14.00` | 540 |
| `6.00` | 483 |
| `20.00` | 477 |
| `8.00` | 460 |
| `15.00` | 367 |
| `12.00` | 255 |
| `9.00` | 180 |
| `21.00` | 176 |
| `30.00` | 113 |
| `25.00` | 101 |
| `50.00` | 100 |
| `28.00` | 54 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Stable question since early codebase. Most recent change was JSON formatting in 2020.

## Notes
Continuous dietary variable tracking weekly sugar-sweetened beverage consumption.
