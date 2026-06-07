# CAD

**Benchmark column**: `CAD`
**Raw identifier**: Derived from `heart_disease` enumeration options (values 1, 3, 5)
**Role**: target
**Type**: binary

## Source
- Derivation: Binary flag; true if any of the coronary artery disease-related options (MI, angina, coronary blockage) are selected in the `heart_disease` multi-select field
- iOS calculation: None — computed post-hoc in MHC-benchmark repo
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `heart_disease` (multi-select enumeration)

## Question
Not directly asked as a single yes/no question — derived from multiple enumeration options within a multi-select survey. Users answer: **"Have you been diagnosed with any of the below diseases?"** and may select any combination of coronary artery disease-related diagnoses (see cardio_CVhealth_survey.json, lines 83–153).

## Derivation details

The `CAD` (Coronary Artery Disease) flag is set to **true (binary 1)** if **any** of the following values are present in the `heart_disease` field:
- Value **1**: "Heart Attack/Myocardial Infarction"
- Value **3**: "Coronary Blockage/Stenosis"
- Value **5**: "Angina (heart-related chest pains)"

**Source enumerations** (from cardio_CVhealth_survey.json):
```json
{
  "label": "Heart Attack/Myocardial Infarction ",
  "value": 1,
  "type": "SurveyQuestionOption"
},
{
  "label": "Coronary Blockage/Stenosis",
  "value": 3,
  "type": "SurveyQuestionOption"
},
{
  "label": "Angina (heart-related chest pains)",
  "value": 5,
  "type": "SurveyQuestionOption"
}
```

If none of these values are selected (or if the user only selects unrelated heart disease diagnoses or "None of the above", value 10), the flag is **false (binary 0)**.

## Observed values

**Total observations**: 30,019 — **type-enforced**: 30,019 (**unique**: 2) — raw Python types seen: `bool` (30,019).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 29,049 | 96.8% |
| `True` | 970 | 3.2% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Survey definition: `cardio_CVhealth_survey.json`
  - `c1833d4` [MHC-780] Add PAH response to vascular survey
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support
  - Earlier history shows multiple survey refinements

## Notes
- **Binary type**: true (1) if the user reports any of the three CAD-related diagnoses; false (0) otherwise.
- This is a composite flag combining Myocardial Infarction, Coronary Blockage, and Angina—the key clinical manifestations of Coronary Artery Disease.
- Related procedures (Coronary Stent/Angioplasty, Heart Bypass Surgery, High Coronary Calcium Score) are NOT included in this CAD flag, though they often co-occur with the diagnoses listed above.
- Part of the broader `heart_disease` multi-select question.
- Cross-reference: see `cardiovascular_disease.md` for the umbrella disease flag, and other disease-specific files for Heart Failure, Afib, etc.
