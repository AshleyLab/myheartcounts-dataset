# chronic_illness_impact

**Benchmark column**: `field_chronic_illness_impact`
**Raw identifier**: `chronic_illness_impact`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json`
- Line: 61
- Survey: `Illness_mindset_inventory` (Illness Mindset Inventory / IMI)

## Question
> Chronic illness negatively impacts nearly all parts of life.

## Answer options
| Value | Label |
|-------|-------|
| 1 | Strongly Agree |
| 2 | Agree |
| 3 | Somewhat Agree |
| 4 | Somewhat Disagree |
| 5 | Disagree |
| 6 | Strongly Disagree |

## Observed values

**Total observations**: 1,009 — **type-enforced**: 1,009 (**unique**: 6) — raw Python types seen: `float` (1,009).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 347 | 34.4% |
| `1` | 286 | 28.3% |
| `3` | 212 | 21.0% |
| `5` | 65 | 6.4% |
| `4` | 64 | 6.3% |
| `6` | 35 | 3.5% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3
- Recent material change: 5043960 (2019-01-23) MHC-610 Add Illness Mindset Inventory Survey
- Notes: Initial addition of survey; no targeted changes since

## Notes
Item 2 from the Illness Mindset Inventory (IMI), a 21-item battery assessing beliefs about chronic illness and body self-healing capacity. All items use a 6-point agreement scale (Strongly Agree to Strongly Disagree).
