# Cerebrovascular Disease

**Benchmark column**: `Cerebrovascular Disease`
**Raw identifier**: Derived from `vascular` enumeration options (values 1, 2)
**Role**: target
**Type**: binary

## Source
- Derivation: Binary flag; true if either Stroke or TIA option is selected in the `vascular` multi-select field
- iOS calculation: None — computed post-hoc in MHC-benchmark repo
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `vascular` (multi-select enumeration)

## Question
Not directly asked as a single yes/no question — derived from multiple enumeration options within a multi-select survey. Users answer: **"Which vascular disease diagnosis have you received?"** and may select any combination of vascular diagnoses, including stroke or TIA (see cardio_CVhealth_survey.json, lines 156–211).

## Derivation details

The `Cerebrovascular Disease` flag is set to **true (binary 1)** if **any** of the following values are present in the `vascular` field:
- Value **1**: "Stroke"
- Value **2**: "Transient Ischemic Attack (TIA)"

**Source enumerations** (from cardio_CVhealth_survey.json):
```json
{
  "label": "Stroke",
  "value": 1,
  "type": "SurveyQuestionOption"
},
{
  "label": "Transient Ischemic Attack (TIA)",
  "value": 2,
  "type": "SurveyQuestionOption"
}
```

If neither value is selected (or if the user selects unrelated vascular diagnoses or "None of the above", value 7), the flag is **false (binary 0)**.

## Observed values

**Total observations**: 30,019 — **type-enforced**: 30,019 (**unique**: 2) — raw Python types seen: `bool` (30,019).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 29,548 | 98.4% |
| `True` | 471 | 1.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Survey definition: `cardio_CVhealth_survey.json`
  - `c1833d4` [MHC-780] Add PAH response to vascular survey
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support

## Notes
- **Binary type**: true (1) if the user reports any cerebrovascular disease (stroke or TIA); false (0) otherwise.
- This is a composite flag combining Stroke and TIA—the key cerebrovascular events affecting the brain.
- Stroke represents an acute ischemic or hemorrhagic event; TIA is a transient ischemic event ("warning stroke").
- Related carotid interventions (Carotid Artery Blockage/Stenosis, Carotid Artery Surgery or Stent) are NOT included in this cerebrovascular flag, though they may co-occur with stroke risk.
- Part of the broader `vascular` multi-select question.
- Cross-reference: see `cardiovascular_disease.md` for the umbrella disease flag, and `Peripheral-Systemic Vascular Disease.md` for peripheral vascular diagnoses.
