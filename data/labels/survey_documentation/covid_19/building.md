# building

**Benchmark column**: `field_building`
**Raw identifier**: `building` (also constant `kStepBuilding` in `CardioHealth/TasksAndSteps/Covid-19/APHCovid19Task.m:36`)
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json`
- Line: 831
- Survey: COVID-19 Survey (`Covid_19_survey`)

## Question
> How many people live in your building?

## Answer options
| Value | Label |
|-------|-------|
| 1 | 1-5 |
| 2 | 6-20 |
| 3 | 21-100 |
| 4 | > 100 |

Data type: integer. `allowMultiple: false`, `allowOther: false`. UI hint: `list`.

## Observed values

**Total observations**: 1,026 — **type-enforced**: 1,026 (**unique**: 4) — raw Python types seen: `float` (1,026).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 847 | 82.6% |
| `2` | 73 | 7.1% |
| `3` | 54 | 5.3% |
| `4` | 52 | 5.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
COVID-19 survey file commits (recent): `d4ea9b8` MHC-791 (update), `fc032df` MHC-775 (Diarrhoea→Diarrhea), `66d17e9` MHC-775 (outro text), `9d527c0` MHC-775 (add exposure questions), `0716733` MHC-775 (study drug), `bacca76` MHC-775 (ACE/ARB), `9ecb11f` MHC-775 (healthcare worker).

## Notes
- Household/building density question — an exposure-context item alongside `exposure`, `self_isolating`, `face_covering` in the COVID survey.
- Missed in the original documentation pass because the COVID agent was scoped to items that looked health-outcome-related; `building` looked like a demographic and was skipped.
