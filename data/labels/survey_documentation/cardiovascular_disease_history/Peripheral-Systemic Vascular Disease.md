# Peripheral/Systemic Vascular Disease

**Benchmark column**: `Peripheral/Systemic Vascular Disease`
**Raw identifier**: Derived from `vascular` enumeration option (value 5)
**Role**: target
**Type**: binary

## Source
- Derivation: Binary flag; true if option value 5 is selected in the `vascular` multi-select field
- iOS calculation: None — computed post-hoc in MHC-benchmark repo
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `vascular` (multi-select enumeration)

## Question
Not directly asked as a yes/no question — derived from a multi-select survey option. Users answer: **"Which vascular disease diagnosis have you received?"** and select from enumeration options including "Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)" (see cardio_CVhealth_survey.json, lines 156–211).

## Derivation details

The `Peripheral-Systemic Vascular Disease` flag is set to **true (binary 1)** if the enumeration value **5** is present in the `vascular` field, which corresponds to the option label **"Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)"**.

**Source enumeration** (from cardio_CVhealth_survey.json, lines 183–187):
```json
{
  "label": "Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)",
  "value": 5,
  "type": "SurveyQuestionOption"
}
```

If value 5 is not selected (or if the user selects "None of the above", value 7), the flag is **false (binary 0)**.

## Observed values

**Total observations**: 30,019 — **type-enforced**: 30,019 (**unique**: 2) — raw Python types seen: `bool` (30,019).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 29,856 | 99.5% |
| `True` | 163 | 0.5% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Survey definition: `cardio_CVhealth_survey.json`
  - `c1833d4` [MHC-780] Add PAH response to vascular survey
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support

## Notes
- **Binary type**: true (1) if the user reports Peripheral Vascular Disease; false (0) otherwise.
- This flag captures narrowing or blockage of blood vessels outside the heart and brain, including the legs, arms, and other organs.
- The label encompasses blockage/stenosis events and interventions (surgery, stent placement).
- Part of the broader `vascular` multi-select question.
- Cross-reference: see `cardiovascular_disease.md` for the umbrella disease flag, and `Cerebrovascular Disease.md` for stroke/TIA flags.
- **Filename note**: The display name "Peripheral/Systemic Vascular Disease" uses a forward slash, which is not allowed in filenames; the filename uses a dash instead: `Peripheral-Systemic Vascular Disease.md`. The exact display name is preserved in this document header.
