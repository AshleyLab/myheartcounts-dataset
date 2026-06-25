# FitzpatrickSkinType

**Benchmark column**: `field_FitzpatrickSkinType`
**Raw identifier**: `HKCharacteristicTypeIdentifierFitzpatrickSkinType` (HealthKit) / `kAPCUserInfoItemTypeFitzpatrickSkinType` (AppCore user-info)
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1330 (HealthKit characteristic registration) and 1432 (AppCore user-info item registration)
- Collected via: HealthKit characteristic read + AppCore profile — not a survey question.

## Question
Not a survey variable. The user sets their Fitzpatrick skin type in the iOS Health app; the MHC app reads it via HealthKit (`HKCharacteristicTypeIdentifierFitzpatrickSkinType`) and/or during AppCore profile setup (`kAPCUserInfoItemTypeFitzpatrickSkinType`).

This is the standard 6-category dermatological classification of skin response to UV exposure (Fitzpatrick Skin Type, codes I-VI). Earlier doc revisions misspelled this as "FrickSkinType"; the actual benchmark column is `field_FitzpatrickSkinType`.

## Answer options
Standard Fitzpatrick scale (HKFitzpatrickSkinType enum values):

| Value | Label | Description |
|-------|-------|-------------|
| 1 | Type I | Pale white skin; always burns, never tans |
| 2 | Type II | White skin; usually burns, tans minimally |
| 3 | Type III | White to light brown; sometimes mild burn, tans uniformly |
| 4 | Type IV | Moderate brown; rarely burns, tans well |
| 5 | Type V | Dark brown; very rarely burns, tans very easily |
| 6 | Type VI | Deeply pigmented dark brown/black; never burns |
| 0 | NotSet | User has not set their skin type in the Health app |

## Observed values

**Total observations**: 228 — **type-enforced**: 228 (**unique**: 3) — raw Python types seen: `str` (228).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (medium) | 129 | 56.6% |
| `0` (light) | 91 | 39.9% |
| `2` (dark) | 8 | 3.5% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- `APHAppDelegate.m` has 117+ commits. The Fitzpatrick characteristic has been registered since early app history; no targeted changes noted.

## Notes
- The benchmark column is `field_FitzpatrickSkinType`. Earlier doc revisions used the misspelling "FrickSkinType" (since corrected in 2026-04-27).
- Read via the HealthKit characteristic API on a one-shot basis (no longitudinal delivery since this is a user-reported trait).
- Released bucketed: Type V (n=4) and Type VI (n=3) were uniquely identifying, so the released artifact ships `{light, medium, dark}` (ordinal).
