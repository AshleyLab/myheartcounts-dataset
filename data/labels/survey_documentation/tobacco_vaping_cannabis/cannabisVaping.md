# cannabisVaping

**Benchmark column**: `field_cannabisVaping`
**Raw identifier**: `cannabisVaping`
**Role**: context
**Type**: ordinal (yes / past / never gating question)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: 1048
- Survey: Vaping and Smoking Survey

## Question
> Do you vape cannabis or cannabis containing products?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes, currently |
| 2 | No, but I have in the past |
| 3 | No, I never have |

Data type: integer. `allowMultiple: false`, `allowOther: false`. UI hint: `list`.

### Skip logic
- Answering `3` (never) → `END_OF_SURVEY` (skips all downstream cannabis-vaping items).
- Answering `2` (past) → skip to `pastCannabisVaping`.
- Answering `1` (currently) → falls through to `currentCannabisVaping`.

## Observed values

**Total observations**: 738 — **type-enforced**: 738 (**unique**: 3) — raw Python types seen: `float` (738).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 616 | 83.5% |
| `2` | 93 | 12.6% |
| `1` | 29 | 3.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
File has 9 commits. Recent: `9f9e14a` MHC-780, `b190fb1` generic update, `c4ec4cf` MHC-772, `2996aab` MHC-756 (update vaping & smoking), `81019db` MHC-710 (add "None of the above"), `a506ed9` MHC-614 (initial Vaping & Smoking Survey).

## Notes
- **Distinct from `cannabisSmoking`** (in `tobacco_vaping_cannabis/`), which asks about smoking cannabis, not vaping it.
- This is the gating question for a 4-item cannabis-vaping battery: `currentCannabisVaping`, `durationCannabisVaping` (not in benchmark spec), `pastCannabisVaping`, `lastCannabisVaping`.
- Missed in the original documentation pass — agents worked from the user-provided "17 smoking/vaping/cannabis history items" count that didn't include the cannabis-vaping branch.
