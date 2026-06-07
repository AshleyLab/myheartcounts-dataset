# fruit

**Benchmark column**: `field_fruit`
**Raw identifier**: `fruit`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~8-18
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How many cups of fruit do you eat in an average day?

## Answer options
**Data type**: decimal  
**Unit**: cups  
**Range**: 0 to 50  
**Input**: numberfield

## Observed values

**Total observations**: 29,530 — **type-enforced**: 29,530 (**unique**: 31) — raw Python types seen: `float` (29,530).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 1.00 |
| median | 1.00 |
| mean | 1.37 |
| q75 | 2.00 |
| max | 50.00 |
| std | 1.49 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.00` | 12,097 |
| `0` | 6,569 |
| `2.00` | 6,351 |
| `3.00` | 2,454 |
| `4.00` | 824 |
| `5.00` | 503 |
| `0.50` | 351 |
| `6.00` | 117 |
| `1.50` | 54 |
| `8.00` | 43 |
| `0.25` | 38 |
| `10.00` | 22 |
| `2.50` | 17 |
| `7.00` | 17 |
| `0.20` | 16 |
| `0.10` | 8 |
| `0.30` | 8 |
| `50.00` | 8 |
| `9.00` | 6 |
| `15.00` | 6 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Stable question since early codebase. Most recent change was JSON formatting in 2020.

## Notes
Continuous dietary variable tracking daily fruit consumption in cups.
