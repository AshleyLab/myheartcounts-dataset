# grains

**Benchmark column**: `field_grains`
**Raw identifier**: `grains`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~50-60
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How many servings of whole grains do you eat on an average day?

## Answer options
**Data type**: integer  
**Unit**: servings  
**Range**: 0 to 50  
**Input**: numberfield

## Observed values

**Total observations**: 29,190 — **type-enforced**: 29,190 (**unique**: 28) — raw Python types seen: `float` (29,190).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 1.00 |
| median | 2.00 |
| mean | 2.20 |
| q75 | 3.00 |
| max | 50.00 |
| std | 2.33 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.00` | 9,303 |
| `2.00` | 7,748 |
| `3.00` | 3,987 |
| `0` | 3,374 |
| `4.00` | 1,949 |
| `5.00` | 1,486 |
| `6.00` | 412 |
| `7.00` | 332 |
| `10.00` | 201 |
| `8.00` | 176 |
| `12.00` | 41 |
| `20.00` | 39 |
| `14.00` | 31 |
| `15.00` | 30 |
| `9.00` | 27 |
| `50.00` | 14 |
| `25.00` | 9 |
| `30.00` | 7 |
| `11.00` | 6 |
| `18.00` | 4 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Stable question since early codebase. Most recent change was JSON formatting in 2020.

## Notes
Continuous dietary variable tracking daily whole grains servings consumption.
