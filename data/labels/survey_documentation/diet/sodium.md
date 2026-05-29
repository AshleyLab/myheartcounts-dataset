# sodium

**Benchmark column**: `field_sodium`
**Raw identifier**: `sodium`
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json`
- Line: ~78-109
- Survey: `Diet_survey_cardio_SchemaV2`

## Question
> Select the statements that apply to you:

## Answer options
| Value | Label |
|-------|-------|
| 1 | I avoid eating prepackaged and processed foods. |
| 2 | I avoid eating out, but when I do, I seek out low-sodium options. |
| 3 | I avoid salt when I'm cooking at home. |
| 4 | None of the above |

**Data type**: integer  
**Input**: list (checkbox, allowMultiple: true)  
**Special**: Value 4 has ignoreOthers flag (mutually exclusive)

## Observed values

**Total observations**: 27,324 — **type-enforced**: 27,324 (**unique**: 13) — raw Python types seen: `list` (27,324).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 13 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(1)` | 8,380 |
| `(4)` | 4,950 |
| `(3)` | 4,906 |
| `(1, 2, 3)` | 2,634 |
| `(1, 3)` | 2,580 |
| `(2)` | 1,703 |
| `(1, 2)` | 1,350 |
| `(2, 3)` | 737 |
| `(1, 4)` | 46 |
| `(3, 4)` | 22 |
| `(2, 4)` | 13 |
| `(2, 3, 4)` | 2 |
| `(1, 2, 4)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 14,991 |
| 2 |  | 6,440 |
| 3 |  | 10,881 |
| 4 |  | 5,034 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 8 (since 2015-08-11)
- Recent material change: a244471 2020-04-03 [MHC-756] Format diet survey json file
- Notes: Ordinal multi-select with smart "None of the above" behavior (ignoreOthers flag). Evolved over multiple commits for ResearchKit 2.0 support and improved multi-choice handling.

## Notes
Multi-select ordinal variable assessing sodium intake behaviors. The "None of the above" option (value 4) is marked with ignoreOthers, enabling smart UI that deselects other options if selected, as per commit 76292e2 (2017) and b447344 (2019).
