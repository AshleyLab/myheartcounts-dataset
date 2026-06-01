# chronic_illness_handling

**Benchmark column**: `field_chronic_illness_handling`
**Raw identifier**: `chronic_illness_handling`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json`
- Line: 376
- Survey: `Illness_mindset_inventory` (Illness Mindset Inventory / IMI)

## Question
> A chronic illness is something that can be dealt with.

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

**Total observations**: 1,006 — **type-enforced**: 1,006 (**unique**: 6) — raw Python types seen: `float` (1,006).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 446 | 44.3% |
| `3` | 341 | 33.9% |
| `1` | 158 | 15.7% |
| `4` | 45 | 4.5% |
| `5` | 12 | 1.2% |
| `6` | 4 | 0.4% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3
- Recent material change: 5043960 (2019-01-23) MHC-610 Add Illness Mindset Inventory Survey
- Notes: Initial addition of survey; no targeted changes since

## Notes
Item 9 from the Illness Mindset Inventory (IMI), a 21-item battery assessing beliefs about chronic illness and body self-healing capacity. All items use a 6-point agreement scale (Strongly Agree to Strongly Disagree).
