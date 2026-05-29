# atwork

**Benchmark column**: `field_atwork`
**Raw identifier**: `atwork`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~28
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Work Time Activity.

**Prompt Detail**: During the past month, which statement best describes the kinds of physical activity you usually did as part of your job or work? Do not include activities during breaks or outside of work. Please read all four statements before selecting one.

## Answer options
Ordinal single-select (enumeration).

| Value | Label | Detail |
|-------|-------|--------|
| 1 | I spent most of the day sitting or standing | When I was at work, I did such things as writing, typing, talking on the telephone, assembling small parts, or operating a machine that takes very little exertion or strength. If I drove a car or truck while at work, I did not lift or carry anything for more that a few minutes each day. |
| 2 | I spent most of the day walking or using my hands and arms in work that required moderate exertion. | When I was at work, I did such things as delivering mail, patrolling on guard duty, doing mechanical work on automobiles or other large machines, house painting, or operating a machine that requires some moderate-activity work of me. If I drove a truck or lift, my job required me to lift and carry things frequently. |
| 3 | I spent most of the day lifting or carrying heavy objects or moving most of my body in some other way. | When I was at work, I did such things as stacking cargo or inventory, handling parts or materials, or doing work like that of a carpenter who builds structures or a gardener who does most of the work without machines. |
| 4 | I spent most of the day doing hard physical labor. | When I was at work, I did such things as digging or chopping with heavy tools or carrying heavy loads (bricks, for example) to the place where they were to be used. If I drove a truck or operated equipment, my job also required me to do hard physical work most of the day with only short breaks. |

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Multiple selections | No |
| Allow other | No |
| UI Hint | list |

## Observed values

**Total observations**: 37,066 — **type-enforced**: 37,066 (**unique**: 4) — raw Python types seen: `str` (37,066).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` | 27,887 | 75.2% |
| `2` | 7,835 | 21.1% |
| `3` | 1,067 | 2.9% |
| `4` | 277 | 0.7% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Conditional Logic
- **Shown only if**: `work == 1` (respondent answered "Yes" to regular work).
- **Skip logic**: None; this is a required question when shown.

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Conditional on `work` (target, binary): only shown if user answers "Yes" to regular work.
- Provides categorical context for occupational activity level: ranges from sedentary (1) to hard labor (4).
- Complements objective activity tracking from HealthKit/CoreMotion during work hours.
- Used to stratify activity analysis by occupation type and helps interpret daily activity patterns.
- Important covariate in cardiovascular health models: occupational activity significantly influences total daily activity and health outcomes.
