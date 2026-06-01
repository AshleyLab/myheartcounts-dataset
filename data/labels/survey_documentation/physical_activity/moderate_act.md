# moderate_act

**Benchmark column**: `field_moderate_act`
**Raw identifier**: `moderate_act`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~122
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Overall, how many minutes of moderate activity do you get in a week?

**Detail**: A person doing moderate-intensity activity, such as a brisk walk, can usually talk but not sing during the activity.

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

**Total observations**: 42,578 — **type-enforced**: 42,578 (**unique**: 409) — raw Python types seen: `float` (42,578).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 35.00 |
| median | 90.00 |
| mean | 4.706e+13 |
| q75 | 200 |
| max | 2.004e+18 |
| std | 9.711e+15 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 5,775 |
| `120` | 4,148 |
| `30.00` | 3,674 |
| `90.00` | 2,258 |
| `180` | 2,236 |
| `300` | 1,986 |
| `150` | 1,685 |
| `200` | 1,619 |
| `0` | 1,394 |
| `100` | 1,303 |
| `20.00` | 1,176 |
| `240` | 1,097 |
| `10.00` | 909 |
| `45.00` | 891 |
| `15.00` | 730 |
| `210` | 694 |
| `5.00` | 674 |
| `360` | 597 |
| `420` | 575 |
| `250` | 480 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Paired with `vigorous_act` (target, continuous) to measure overall weekly activity intensity distribution.
- Complementary measure: moderate-intensity activity is the lower-intensity component of MVPA (Moderate-Vigorous Physical Activity).
- Used alongside `phys_activity` (leisure-time activity categorical response) to cross-validate activity level.
- Contributes to cardiovascular health assessment: moderate activity also provides health benefits but is distinct from vigorous activity in intensity.
- Both `moderate_act` and `vigorous_act` together inform personalized coaching recommendations.
