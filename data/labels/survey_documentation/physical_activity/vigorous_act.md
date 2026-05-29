# vigorous_act

**Benchmark column**: `vigorous_act`
**Raw identifier**: `vigorous_act`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~137
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Overall, how many minutes of vigorous activity do you get in a week?

**Detail**: A person doing vigorous-intensity activity, such as running, usually cannot say more than a few words without pausing for a breath.

## Answer options
Continuous integer input (slider).

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Min Value | 0 |
| Max Value | 2000 |
| Unit | minutes per week |
| UI Hint | slide |

## Observed values

**Total observations**: 42,497 — **type-enforced**: 42,497 (**unique**: 256) — raw Python types seen: `float` (42,497).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 2.00 |
| median | 30.00 |
| mean | 73.89 |
| q75 | 90.00 |
| max | 1440 |
| std | 136.8 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `0` | 9,700 |
| `30.00` | 4,705 |
| `60.00` | 4,128 |
| `10.00` | 2,610 |
| `20.00` | 2,292 |
| `120` | 2,106 |
| `15.00` | 1,715 |
| `5.00` | 1,558 |
| `90.00` | 1,334 |
| `180` | 1,240 |
| `100` | 964 |
| `300` | 786 |
| `150` | 778 |
| `45.00` | 752 |
| `200` | 714 |
| `240` | 519 |
| `1.00` | 501 |
| `2.00` | 495 |
| `50.00` | 482 |
| `40.00` | 473 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Paired with `moderate_act` (context variable) to measure overall weekly activity intensity distribution.
- Used in coaching logic to determine activity level and personalized interventions.
- No derived targets depend directly on vigorous_act; however, vigorous activity is often weighted more heavily in cardiovascular health assessments.
- Related to `phys_activity` (leisure time activity ordinal response) which provides categorical context.
