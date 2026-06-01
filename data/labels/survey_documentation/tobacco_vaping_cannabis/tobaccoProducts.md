# tobaccoProducts

**Benchmark column**: `field_tobaccoProducts`
**Raw identifier**: `tobaccoProducts`
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~699
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Which of the following products have you used in the past week?

## Answer options
| Value | Label | Special |
|-------|-------|---------|
| 1 | Manufactured cigarettes | |
| 2 | Hand rolled cigarettes | |
| 3 | Pipe full of tobacco | |
| 4 | Cigars, cigarillos | |
| 5 | Water Pipe (hookah) | |
| 6 | E-cigarettes | |
| 7 | Chew tobacco | |
| 8 | None of the above | ignoreOthers=true |

## Observed values

**Total observations**: 970 — **type-enforced**: 970 (**unique**: 32) — raw Python types seen: `list` (970).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 32 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(8)` | 793 |
| `(1)` | 45 |
| `(6)` | 44 |
| `(1, 6)` | 14 |
| `(7)` | 11 |
| `(4)` | 10 |
| `(1, 2)` | 10 |
| `(2)` | 6 |
| `(5)` | 5 |
| `(1, 4)` | 4 |
| `(1, 4, 6)` | 3 |
| `(1, 2, 6)` | 2 |
| `(2, 6)` | 2 |
| `(1, 2, 4, 6)` | 2 |
| `(1, 5, 6)` | 2 |
| `(4, 6)` | 1 |
| `(1, 3, 5)` | 1 |
| `(1, 3, 7)` | 1 |
| `(4, 6, 7)` | 1 |
| `(3)` | 1 |
| `(1, 6, 7)` | 1 |
| `(1, 2, 4, 5, 6, 7)` | 1 |
| `(1, 3)` | 1 |
| `(3, 5)` | 1 |
| `(1, 2, 5)` | 1 |
| `(1, 4, 7)` | 1 |
| `(1, 3, 4, 7)` | 1 |
| `(1, 4, 5, 6, 7)` | 1 |
| `(1, 2, 4)` | 1 |
| `(3, 6)` | 1 |
| `(2, 5, 6)` | 1 |
| `(1, 7)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 93 |
| 2 |  | 26 |
| 3 |  | 7 |
| 4 |  | 26 |
| 5 |  | 13 |
| 6 |  | 76 |
| 7 |  | 19 |
| 8 |  | 793 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Multi-select option added (allowMultiple: true) in MHC-710; "None of the above" option with ignoreOthers flag added

## Notes
Multi-select question; participants can choose multiple products. "None of the above" (value 8) has ignoreOthers flag set to true, which should disable other selections if chosen. Past week timeframe is relatively short.
