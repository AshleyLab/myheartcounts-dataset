# sleep_diagnosis2

**Benchmark column**: `field_sleep_diagnosis2`
**Raw identifier**: `sleep_diagnosis2`
**Role**: context
**Type**: multi_categorical

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~200
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Which of the following sleep disorders apply to you? (select all that apply).

## Answer options
Categorical multi-select (enumeration). Respondents may select multiple options; option 8 ("None of the above") has `ignoreOthers: true`, meaning it deselects other choices if selected.

| Value | Label |
|-------|-------|
| 1 | Sleep Apnea (breathing stops at night) |
| 2 | Insomnia |
| 3 | Circadian Rhythm Disturbance (Advanced/Delayed Sleep Phase, Shift work) |
| 4 | Restless Legs Syndrome and/or Periodic Limb Movements |
| 5 | Narcolepsy or Cataplexy |
| 6 | REM Behavior Disorder (act out dreams in your sleep) |
| 7 | Sleepwalking |
| 8 | None of the above |

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Multiple selections | Yes |
| Allow other | No |
| Exclusive option (8) | Yes – "None of the above" deselects others |
| UI Hint | list |

## Observed values

**Total observations**: 348 — **type-enforced**: 348 (**unique**: 39) — raw Python types seen: `list` (348).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

**Top 39 selections (sorted code tuples)**:

| selection | count |
|-----------|------:|
| `(1)` | 175 |
| `(2)` | 48 |
| `(8)` | 22 |
| `(1, 2)` | 21 |
| `(1, 4)` | 10 |
| `(1, 2, 4)` | 10 |
| `(2, 4)` | 10 |
| `(5)` | 5 |
| `(2, 3)` | 4 |
| `(3)` | 3 |
| `(2, 3, 4)` | 3 |
| `(4)` | 3 |
| `(1, 2, 3, 4)` | 2 |
| `(1, 2, 3)` | 2 |
| `(2, 6)` | 2 |
| `(1, 2, 4, 6)` | 2 |
| `(1, 3)` | 2 |
| `(1, 2, 6, 7)` | 2 |
| `(1, 6)` | 2 |
| `(3, 4, 5, 6, 7)` | 1 |
| `(1, 2, 3, 4, 7)` | 1 |
| `(1, 4, 7)` | 1 |
| `(1, 5, 7)` | 1 |
| `(1, 2, 4, 6, 7)` | 1 |
| `(1, 5, 6)` | 1 |
| `(1, 7)` | 1 |
| `(1, 2, 3, 5)` | 1 |
| `(1, 2, 7)` | 1 |
| `(5, 6)` | 1 |
| `(4, 5)` | 1 |
| `(6)` | 1 |
| `(1, 2, 3, 4, 6)` | 1 |
| `(2, 4, 5)` | 1 |
| `(2, 5)` | 1 |
| `(1, 2, 3, 6)` | 1 |
| `(1, 2, 4, 5, 6)` | 1 |
| `(2, 5, 7)` | 1 |
| `(4, 6, 7)` | 1 |
| `(1, 4, 5)` | 1 |

**Per-code marginals (a row per option code; users can select multiple)**:

| code | label | count |
|-----:|-------|------:|
| 1 |  | 240 |
| 2 |  | 116 |
| 3 |  | 21 |
| 4 |  | 50 |
| 5 |  | 16 |
| 6 |  | 17 |
| 7 |  | 11 |
| 8 |  | 22 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Conditional Logic
- **Shown only if**: `sleep_diagnosis1 == 1` (respondent answered "Yes" to ever being diagnosed with a sleep disorder).
- **Skip logic**: If `sleep_diagnosis1 == 0`, survey skips to `END_OF_SURVEY` and this question is not shown.

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `b447344` (MHC-626 Fix support for ignoreOthers property)
- Notes: `b447344` specifically added support for the `ignoreOthers` property used by option 8 in this question.

## Notes
- Conditional on `sleep_diagnosis1` (target, binary): only shown if user reports prior clinical sleep disorder diagnosis.
- Multi-select allows capturing comorbid sleep disorders: a respondent may have both sleep apnea and insomnia, for example.
- Special handling: option 8 ("None of the above") is mutually exclusive—selecting it clears other selections and vice versa.
- Clinical significance: sleep disorders have strong cardiovascular implications. Sleep apnea (1), insomnia (2), and circadian rhythm disorders (3) are most common and impactful.
- Used for clinical stratification and to identify high-risk subgroups requiring interventions.
- Complements sleep duration measures (`sleep_time`, `sleep_time1`) and symptom-based coaching triggers.
