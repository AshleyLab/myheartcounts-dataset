# fish

**Benchmark column**: `field_fish`
**Raw identifier**: `fish`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~36-46
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How many servings of fish do you eat on an average week?

## Answer options
**Data type**: integer  
**Unit**: servings  
**Range**: 0 to 50  
**Input**: numberfield

## Observed values

**Total observations**: 29,554 — **type-enforced**: 29,554 (**unique**: 23) — raw Python types seen: `float` (29,554).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 0 |
| median | 1.00 |
| mean | 1.22 |
| q75 | 2.00 |
| max | 50.00 |
| std | 1.52 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.00` | 10,343 |
| `0` | 10,037 |
| `2.00` | 5,345 |
| `3.00` | 2,204 |
| `4.00` | 811 |
| `5.00` | 485 |
| `6.00` | 116 |
| `7.00` | 94 |
| `10.00` | 36 |
| `8.00` | 35 |
| `9.00` | 9 |
| `14.00` | 9 |
| `12.00` | 7 |
| `13.00` | 4 |
| `15.00` | 4 |
| `50.00` | 4 |
| `11.00` | 2 |
| `22.00` | 2 |
| `25.00` | 2 |
| `30.00` | 2 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Stable question since early codebase. Most recent change was JSON formatting in 2020.

## Notes
Continuous dietary variable tracking weekly fish servings consumption.
