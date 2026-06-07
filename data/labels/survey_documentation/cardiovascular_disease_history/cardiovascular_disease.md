# cardiovascular_disease

**Benchmark column**: `cardiovascular_disease`
**Raw identifier**: Derived from `heart_disease` and `vascular` multi-select survey fields
**Role**: target
**Type**: binary

## Source
- Derivation: Union flag; true if any `heart_disease` option (except "None of the above") OR any `vascular` option (except "None of the above") is selected
- iOS calculation: None — computed post-hoc in MHC-benchmark repo
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `heart_disease`, `vascular` (both multi-select, enumeration values)

## Question
Not directly asked — derived from two multi-select survey questions:
1. **"Have you been diagnosed with any of the below diseases?"** (`heart_disease`, see cardio_CVhealth_survey.json, lines 83–153)
2. **"Which vascular disease diagnosis have you received?"** (`vascular`, see cardio_CVhealth_survey.json, lines 156–211)

Users can select multiple options or "None of the above" in each question.

## Derivation details

The `cardiovascular_disease` flag is set to **true (binary 1)** if either condition holds:
- At least one `heart_disease` option is selected (excluding the "None of the above" option, value 10)
- At least one `vascular` option is selected (excluding the "None of the above" option, value 7)

Otherwise, it is **false (binary 0)**.

**Heart Disease Options** (values 1–11, excluding "None" value 10):
1. Heart Attack/Myocardial Infarction
2. Heart Bypass Surgery
3. Coronary Blockage/Stenosis
4. Coronary Stent/Angioplasty
5. Angina (heart-related chest pains)
6. High Coronary Calcium Score
7. Heart Failure or CHF
8. Atrial fibrillation (Afib)
9. Congenital Heart Defect
11. Pulmonary Hypertension

**Vascular Options** (values 1–8, excluding "None" value 7):
1. Stroke
2. Transient Ischemic Attack (TIA)
3. Carotid Artery Blockage/Stenosis
4. Carotid Artery Surgery or Stent
5. Peripheral Vascular Disease (Blockage/Stenosis, Surgery, or Stent)
6. Abdominal Aortic Aneurysm
8. Pulmonary Arterial Hypertension

## Observed values

**Total observations**: 30,019 — **type-enforced**: 30,019 (**unique**: 2) — raw Python types seen: `bool` (30,019).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 27,610 | 92.0% |
| `True` | 2,409 | 8.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related survey: `cardio_CVhealth_survey.json`
  - `c1833d4` [MHC-780] Add PAH response to vascular survey
  - `a8e47e1` MHC-734 CVHealth Survey - add PH Support

## Notes
- **Binary type**: true (1) if any cardiovascular or vascular disease diagnosis is reported; false (0) otherwise.
- This is an umbrella variable capturing all reported cardiovascular and vascular diagnoses.
- Specific disease subtype flags (Heart Failure, Afib, CAD, etc.) are documented separately—see the disease-specific files below.
- Cross-reference: see "Heart Failure or CHF.md", "Atrial fibrillation (Afib).md", "CAD.md", "Congenital Heart.md", "Peripheral-Systemic Vascular Disease.md", "PH.md", and "Cerebrovascular Disease.md" for individual disease flags.
