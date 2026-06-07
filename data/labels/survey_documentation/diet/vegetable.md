# vegetable

**Benchmark column**: `field_vegetable`
**Raw identifier**: `vegetable`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~22-32
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> How many cups of vegetables do you eat in an average day?

## Answer options
**Data type**: decimal  
**Unit**: cups  
**Range**: 0 to 50  
**Input**: numberfield

## Observed values

**Total observations**: 29,555 — **type-enforced**: 29,555 (**unique**: 40) — raw Python types seen: `float` (29,555).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 1.00 |
| median | 2.00 |
| mean | 1.92 |
| q75 | 3.00 |
| max | 50.00 |
| std | 1.68 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.00` | 10,888 |
| `2.00` | 8,098 |
| `3.00` | 3,900 |
| `0` | 2,790 |
| `4.00` | 1,779 |
| `5.00` | 1,010 |
| `6.00` | 384 |
| `0.50` | 220 |
| `8.00` | 156 |
| `10.00` | 71 |
| `7.00` | 59 |
| `1.50` | 57 |
| `0.25` | 27 |
| `2.50` | 24 |
| `9.00` | 15 |
| `15.00` | 12 |
| `12.00` | 10 |
| `0.20` | 9 |
| `0.30` | 6 |
| `50.00` | 6 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Stable question since early codebase. Most recent change was JSON formatting in 2020.

## Notes
Continuous dietary variable tracking daily vegetable consumption in cups.
