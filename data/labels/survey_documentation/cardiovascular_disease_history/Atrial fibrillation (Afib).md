# Atrial fibrillation (Afib)

**Benchmark column**: `Atrial fibrillation (Afib)`
**Raw identifier**: Derived from `heart_disease` enumeration option (value 8)
**Role**: target
**Type**: binary

## Source
- Derivation: Binary flag; true if option value 8 is selected in the `heart_disease` multi-select field
- iOS calculation: None — computed post-hoc in MHC-benchmark repo
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `heart_disease` (multi-select enumeration)

## Question
Not directly asked as a yes/no question — derived from a multi-select survey option. Users answer: **"Have you been diagnosed with any of the below diseases?"** and select from enumeration options including "Atrial fibrillation (Afib)" (see cardio_CVhealth_survey.json, lines 83–153).

## Derivation details

The `Atrial fibrillation (Afib)` flag is set to **true (binary 1)** if the enumeration value **8** is present in the `heart_disease` field, which corresponds to the option label **"Atrial fibrillation (Afib)"**.

**Source enumeration** (from cardio_CVhealth_survey.json, lines 125–129):
```json
{
  "label": "Atrial fibrillation (Afib)",
  "value": 8,
  "type": "SurveyQuestionOption"
}
```

If value 8 is not selected (or if the user selects "None of the above", value 10), the flag is **false (binary 0)**.

## Observed values

**Total observations**: 30,019 — **type-enforced**: 30,019 (**unique**: 2) — raw Python types seen: `bool` (30,019).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 29,634 | 98.7% |
| `True` | 385 | 1.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Survey definition: `cardio_CVhealth_survey.json`
  - `c1833d4` [MHC-780] Add PAH response to vascular survey
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support
  - Earlier history shows multiple survey refinements

## Notes
- **Binary type**: true (1) if the user reports a diagnosis of Atrial Fibrillation; false (0) otherwise.
- This flag captures a common cardiac arrhythmia (irregular heart rhythm) that increases stroke risk.
- Part of the broader `heart_disease` multi-select question.
- Cross-reference: see `cardiovascular_disease.md` for the umbrella disease flag, and other disease-specific files for Heart Failure, CAD, etc.
