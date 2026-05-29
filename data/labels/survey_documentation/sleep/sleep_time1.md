# sleep_time1

**Benchmark column**: `field_sleep_time1`
**Raw identifier**: `sleep_time1`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~152
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> How much sleep do you usually get at night on weekdays or workdays?
> (in hours)

## Answer options
Continuous integer input (slider).

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Min Value | 0 |
| Max Value | 24 |
| Unit | hours |
| UI Hint | slide |

## Observed values

**Total observations**: 44,362 — **type-enforced**: 44,362 (**unique**: 127) — raw Python types seen: `float` (44,362).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | -4.571e+18 |
| q25 | 6.00 |
| median | 7.00 |
| mean | -1.03e+14 |
| q75 | 8.00 |
| max | 9.892e+11 |
| std | 2.17e+16 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `7.00` | 16,274 |
| `6.00` | 10,882 |
| `8.00` | 9,870 |
| `5.00` | 3,430 |
| `9.00` | 1,749 |
| `4.00` | 741 |
| `10.00` | 535 |
| `3.00` | 128 |
| `12.00` | 103 |
| `2.00` | 60 |
| `40.00` | 53 |
| `11.00` | 49 |
| `30.00` | 42 |
| `35.00` | 40 |
| `49.00` | 25 |
| `42.00` | 24 |
| `50.00` | 22 |
| `45.00` | 20 |
| `0` | 18 |
| `1.00` | 18 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Measures actual weekday/workday sleep duration (subjective self-report).
- Complements `sleep_time` (perceived sleep need): comparison between perceived need and actual weekday sleep can reveal sleep deficits or satisfaction.
- One of three sleep-related continuous variables: `sleep_time` (perceived need), `sleep_time1` (weekday actual), and potentially a third sleep time variable (not found in this survey).
- Weekday-specific: captures sleep during work/school week, which may differ from weekend sleep and is often more constrained.
- Note: `WakeUpTime` and `GoSleepTime` are NOT in this survey; they may be derived from HealthKit sleep analysis or tracked elsewhere.
- Used to assess sleep quantity: values <7 hours are associated with cardiovascular risk; values >9 hours may indicate sleep apnea or other sleep disorders.
- Informs coaching: respondents with inadequate weekday sleep may receive sleep hygiene recommendations or work-stress management suggestions.
