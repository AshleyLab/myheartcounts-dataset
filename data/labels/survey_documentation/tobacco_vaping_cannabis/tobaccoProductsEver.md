# tobaccoProductsEver

**Benchmark column**: `field_tobaccoProductsEver`
**Raw identifier**: `tobaccoProductsEver`
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~755
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Which of the following products have you ever used?

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

**Total observations**: 1,109 — **type-enforced**: 1,109 (**unique**: 69) — raw Python types seen: `list` (1,109).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 69 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(8)` | 487 |
| `(1)` | 142 |
| `(4)` | 37 |
| `(1, 2)` | 32 |
| `(1, 6)` | 31 |
| `(1, 4)` | 27 |
| `(5)` | 19 |
| `(1, 2, 3, 4, 5, 6, 7)` | 19 |
| `(1, 2, 3, 4)` | 18 |
| `(1, 2, 3, 4, 5, 6)` | 18 |
| `(1, 2, 4, 6)` | 16 |
| `(1, 2, 4)` | 16 |
| `(6)` | 16 |
| `(1, 4, 5)` | 13 |
| `(1, 5)` | 13 |
| `(1, 2, 4, 5, 6)` | 12 |
| `(1, 4, 5, 6)` | 12 |
| `(1, 2, 6)` | 11 |
| `(1, 2, 3, 4, 5)` | 11 |
| `(1, 3, 4)` | 11 |
| `(7)` | 10 |
| `(3)` | 9 |
| `(1, 2, 4, 7)` | 9 |
| `(1, 4, 7)` | 8 |
| `(1, 2, 4, 5, 6, 7)` | 7 |
| `(1, 4, 6)` | 7 |
| `(4, 5)` | 6 |
| `(3, 4)` | 6 |
| `(1, 3, 4, 7)` | 6 |
| `(1, 4, 5, 7)` | 4 |
| `(1, 5, 6)` | 4 |
| `(1, 2, 3, 4, 7)` | 4 |
| `(1, 2, 3, 4, 6, 7)` | 4 |
| `(1, 3, 4, 6)` | 3 |
| `(1, 7)` | 3 |
| `(1, 4, 5, 6, 7)` | 3 |
| `(3, 4, 5)` | 3 |
| `(1, 2, 3, 4, 5, 7)` | 3 |
| `(2)` | 3 |
| `(1, 2, 4, 5)` | 3 |
| `(1, 2, 5, 6)` | 3 |
| `(1, 2, 4, 5, 7)` | 3 |
| `(1, 2, 5)` | 3 |
| `(1, 3, 7)` | 2 |
| `(4, 7)` | 2 |
| `(1, 6, 7)` | 2 |
| `(1, 2, 4, 6, 7)` | 2 |
| `(1, 2, 7)` | 2 |
| `(1, 3)` | 2 |
| `(1, 2, 3)` | 2 |
| `(1, 4, 6, 7)` | 2 |
| `(4, 5, 7)` | 1 |
| `(2, 6)` | 1 |
| `(1, 3, 4, 5, 6, 7)` | 1 |
| `(2, 3)` | 1 |
| `(4, 5, 6)` | 1 |
| `(1, 2, 3, 4, 6)` | 1 |
| `(3, 5)` | 1 |
| `(5, 6)` | 1 |
| `(1, 3, 4, 5)` | 1 |
| `(3, 4, 7)` | 1 |
| `(1, 3, 4, 5, 7)` | 1 |
| `(3, 4, 5, 7)` | 1 |
| `(3, 7)` | 1 |
| `(2, 3, 4, 5, 6, 7)` | 1 |
| `(1, 3, 4, 5, 6)` | 1 |
| `(1, 2, 6, 7)` | 1 |
| `(2, 4, 6)` | 1 |
| `(5, 6, 7)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 499 |
| 2 |  | 207 |
| 3 |  | 132 |
| 4 |  | 306 |
| 5 |  | 170 |
| 6 |  | 182 |
| 7 |  | 104 |
| 8 |  | 487 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Multi-select option (allowMultiple: true); "None of the above" option with ignoreOthers flag added in MHC-710

## Notes
Multi-select question; participants can choose multiple products they have ever used. "None of the above" (value 8) has ignoreOthers flag set to true, which should disable other selections if chosen. Lifetime exposure assessment (as opposed to tobaccoProducts which is past week).
