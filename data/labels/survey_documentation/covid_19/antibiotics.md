# antibiotics

**Benchmark column**: `field_antibiotics`
**Raw identifier**: `antibiotics`
**Role**: context
**Type**: multi_categorical (source is `list<double>`; "Select all that apply")

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: ~633
- Survey: `Covid_19_survey`

## Question
> Are you taking any of the following antibiotics or immune system modulators regularly in the past week? (Select all that apply)

## Answer options
| Value | Label |
|-------|-------|
| 0 | None |
| 1 | Azithromycin |
| 2 | Doxycycline |
| 3 | Other antibiotic (not azithromycin or doxycycline) |
| 4 | Hydrochloroquine (Plaquenil) |
| 5 | Oral corticosteroids (eg prednisone) |
| 6 | Inhaled corticosteroids (eg budesonide, beclamethasone (QVAR, Pulmicort)) |
| 7 | Tocilizumab (Actemra) |
| 8 | Other Disease Modifying Anti-Rheumatic Drugs (e.g., methotrexate, cyclophosphamide, sulfasalazine, etc.) |
| 9 | Other immuno-suppressives (eg tacrolimus, mycophenolate, sirolimus (Prograf, Cellcept, Rapamune)) |
| 10 | Study drug (Investigational agent) |
| 11 | Remdesivir |

## Observed values

**Total observations**: 1,020 — **type-enforced**: 1,020 (**unique**: 20) — raw Python types seen: `list` (1,020).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 20 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(0)` | 954 |
| `(6)` | 13 |
| `(3)` | 10 |
| `(2)` | 10 |
| `(5)` | 5 |
| `(10)` | 4 |
| `(4)` | 4 |
| `(11)` | 3 |
| `(8, 9)` | 2 |
| `(9)` | 2 |
| `(1)` | 2 |
| `(5, 6, 9)` | 2 |
| `(4, 9)` | 2 |
| `(3, 5, 9, 10)` | 1 |
| `(5, 6, 8)` | 1 |
| `(1, 6)` | 1 |
| `(5, 8)` | 1 |
| `(5, 8, 9)` | 1 |
| `(2, 5, 6)` | 1 |
| `(4, 6)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 0 |  | 954 |
| 1 |  | 3 |
| 2 |  | 11 |
| 3 |  | 11 |
| 4 |  | 7 |
| 5 |  | 12 |
| 6 |  | 19 |
| 8 |  | 5 |
| 9 |  | 10 |
| 10 |  | 5 |
| 11 |  | 3 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 20
- Recent change: d4ea9b8 (MHC-791 Update covid survey)
- Notes: COVID-19 treatment and immunomodulatory medication assessment; multiple-select question

## Notes
Appears only in the main COVID survey, not in the recurrent survey. Multiple-choice question allowing selection of multiple medications. Value 0 (None) has `ignoreOthers: true` flag, indicating it is mutually exclusive. Comprehensive assessment of COVID-19 treatments, immunosuppressants, and disease-modifying agents. Conditional follow-up question for study drug (value 10) collects investigational agent name.
