# sleep_time

**Benchmark column**: `field_sleep_time`
**Raw identifier**: `sleep_time`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~167
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> How much sleep do you think you need every night to be rested? 
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

**Total observations**: 44,359 — **type-enforced**: 44,359 (**unique**: 101) — raw Python types seen: `float` (44,359).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 7.00 |
| median | 8.00 |
| mean | 1.55e+14 |
| q75 | 8.00 |
| max | 6.874e+18 |
| std | 3.264e+16 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `8.00` | 21,478 |
| `7.00` | 10,483 |
| `9.00` | 5,245 |
| `6.00` | 3,549 |
| `10.00` | 1,875 |
| `5.00` | 761 |
| `12.00` | 295 |
| `4.00` | 213 |
| `11.00` | 111 |
| `3.00` | 52 |
| `2.00` | 38 |
| `1.00` | 33 |
| `0` | 16 |
| `14.00` | 15 |
| `50.00` | 14 |
| `13.00` | 13 |
| `15.00` | 12 |
| `56.00` | 10 |
| `40.00` | 9 |
| `730` | 8 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Measures perceived sleep need (subjective): "how much sleep do you think you need" rather than actual sleep obtained.
- Complements `sleep_time1` (actual weekday sleep duration): comparison between perceived need and actual sleep quantity can indicate sleep satisfaction/deficit.
- One of three sleep-related continuous variables: `sleep_time` (perceived need), `sleep_time1` (weekday actual), and potentially `sleep_time2` if it exists (not found in this survey).
- Note: `WakeUpTime` and `GoSleepTime` are NOT in this survey; they may be derived from HealthKit sleep analysis or tracked in another survey (see HealthKit integration notes).
- Used to assess sleep hygiene and awareness: large gaps between perceived need and actual sleep may indicate sleep dissatisfaction or insomnia.
- Informs coaching: respondents with high perceived need relative to actual sleep may benefit from sleep hygiene or activity-adjustment recommendations.
