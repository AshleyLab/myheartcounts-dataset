# phys_activity

**Benchmark column**: `field_phys_activity`
**Raw identifier**: `phys_activity`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~69
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Leisure Time Activity.

**Prompt Detail**: During the past month, which statement best describes the kinds of physical activity you usually did? Do not include the time you spent as part of your work or job. Please read all six statements before selecting one.

## Answer options
Ordinal single-select (enumeration).

| Value | Label | Detail |
|-------|-------|--------|
| 1 | I did not do much physical activity. | I mostly did things like watching television, reading, playing cards, or playing computer games. Only occasionally, no more than once or twice a month, did I do anything more active such as going for a walk or playing tennis. |
| 2 | Once or twice a week, I did light activities | Such as getting outdoors on the weekends for an easy walk or stroll. Or once or twice a week, I did chores around the house such as sweeping floors or vacuuming. |
| 3 | About three times a week, I did moderate activities | Such as brisk walking, swimming, or riding a bike for about 15–20 minutes each time. Or about once a week, I did moderately difficult chores such as raking or mowing the lawn for about 45–60 minutes. Or about once a week, I played sports such as softball, basketball, or soccer for about 45–60 minutes. |
| 4 | Almost daily, that is five or more times a week, I did moderate activities | Such as brisk walking, swimming, or riding a bike for 30 minutes or more each time. Or about once a week, I did moderately difficult chores or played sports for 2 hours or more. |
| 5 | About three times a week, I did vigorous activities | Such as running or riding hard on a bike for 30 minutes or more each time. |
| 6 | Almost daily, that is, five or more times a week, I did vigorous activities | Such as running or riding hard on a bike for 30 minutes or more each time. |

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Multiple selections | No |
| Allow other | No |
| UI Hint | list |

## Observed values

**Total observations**: 44,443 — **type-enforced**: 44,443 (**unique**: 6) — raw Python types seen: `str` (44,443).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 12,037 | 27.1% |
| `3` | 10,131 | 22.8% |
| `1` | 7,008 | 15.8% |
| `4` | 6,724 | 15.1% |
| `5` | 4,956 | 11.2% |
| `6` | 3,587 | 8.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Conditional Logic
- **Shown when**: `work == 0` (user answered "No" to regular work) OR after `atwork` response (if employed).
- **Skip logic**: None; this is a required question when shown.

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Ordinal categorical measure of leisure-time physical activity level.
- Complements objective activity metrics (`vigorous_act`, `moderate_act` continuous minutes) with categorical self-assessment of behavior frequency and intensity.
- Provides important context: respondents who report low leisure activity (1-2) may have different HealthKit patterns than those reporting frequent vigorous activity (5-6).
- Used to validate and cross-reference continuous activity data and identify self-awareness vs. objective activity discrepancies.
- Informs coaching: values 1-2 may trigger activity-promotion coaching; values 5-6 may be reinforced or used for advanced training suggestions.
