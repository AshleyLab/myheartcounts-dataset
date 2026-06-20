# CLAUDE.md

## 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop. Name what's confusing.

## 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No silent fallbacks, prefer failure over obscure behavior.
- No backwards compatibility by default, only if user requests explicitely.
- No features beyond what was asked.
- No abstractions for single-use code.
- No “flexibility” that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

## 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- Don't “improve” adjacent code or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice dead code, mention it — don't delete it.

## 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- “Add validation” → “Write tests, then make them pass”
- “Fix the bug” → “Reproduce it in a test, then fix”
- “Refactor X” → “Ensure tests pass before and after”

## Project Overview

**OpenMHC** is a public evaluation API and leaderboard for the MyHeartCounts wearable health benchmark (NeurIPS 2026). It evaluates models across three tracks:
- **Track 1 — Outcome Prediction**: Health classification/regression from weekly (168h) sensor embeddings (33 tasks)
- **Track 2 — Imputation**: Reconstructing masked daily (1440-min) sensor data across 6 masking scenarios
- **Track 3 — Forecasting**: Time-series forecasting of hourly sensor values

## Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint (ruff)
ruff check src/
ruff format src/

# Download dataset for local development
python -c "import openmhc; openmhc.download_dataset(version='xs')"  # ~1.9 GB dev subset
```

Ruff config: 100-char line length, Google docstrings, targets Python 3.10+. Line-length check (E501) is disabled for docstrings.

## Architecture

### Public API vs. Internal Engines

`src/openmhc/` is the **public-facing package**. Internal evaluation logic lives in track-specific modules (`downstream_evaluation/`, `imputation_evaluation/`, `forecasting_evaluation/`), which users never touch.

```
src/
├── openmhc/                  # Public API (pip install target)
│   ├── __init__.py           # evaluate_prediction, evaluate_imputation, evaluate_forecasting
│   ├── _protocols.py         # Duck-typed Encoder/Imputer/Forecaster protocols
│   ├── _evaluate.py          # Evaluation orchestrators + adapters
│   ├── _dataset.py           # Dataset download + path resolution (Dataverse)
│   ├── _data_utils.py        # iter_train_data, iter_split_data, load_sample_metadata
│   ├── _results.py           # PredictionResults, ImputationResults, ForecastingResults
│   └── imputers/             # Reference implementations (mean, LOCF, temporal, etc.)
├── downstream_evaluation/    # Track 1 internals (sklearn classifiers, feature extraction)
├── imputation_evaluation/    # Track 2 internals (masking, metrics, W&B logging)
├── forecasting_evaluation/   # Track 3 internals (Chronos2, AutoARIMA, etc.)
└── labels/                   # Label registry (33 task names, types, validity criteria)
```

### Hydra CLI (Track 2)

Reproducible Track 2 runs are dispatched via the `mhc-impute-eval` console script (declared in `pyproject.toml`). The CLI composes YAML configs at `configs/imputation/` (repo root), validates against the dataclass schema in `src/imputation_evaluation/config.py`, builds the imputer via the registry in `src/imputation_evaluation/hydra/registry.py`, and forwards to the same `run_eval` library entry point as `openmhc.evaluate_imputation`. Public-API users never touch Hydra. See `src/imputation_evaluation/README.md` Part 1.5 for usage. Tracks 1 and 3 have their own CLI surfaces (`scripts/downstream_eval/`, `scripts/run_forecasting_eval.py`).

### Protocol Pattern (Critical)

Models are integrated via **duck typing** — no base class inheritance. The harness uses `inspect.signature()` to detect optional kwargs and forward only what the imputer/encoder/forecaster declares.

Example: an `Imputer.impute(data, observed_mask, target_mask)` method may optionally accept `user_ids`, `dates`, or `sample_indices` kwargs, which the harness will pass if declared.

### Data Path Resolution

All evaluate/download functions resolve the dataset root in this priority order:
1. Explicit `data_dir=` argument
2. `MHC_DATA_DIR` environment variable
3. Default: `~/.cache/openmhc/data`

### Data Formats

- **Track 1**: Weekly segments (168h) from `daily_hourly_hf/` directory (HuggingFace disk format)
- **Track 2**: Daily segments (1440 min) from `daily_hf/` directory; 19 sensor channels + binary flags
- **Track 3**: Hourly trajectories; requires pre-computed `sample_index_file` JSON (no fallback)
- All tensors: NumPy float32
- Normalization stats: `normalization_stats_hourly.json` (z-score, global)

### Splits & Label Validity

- Canonical split: `sharable_users_seed42_2026.json` (user-level, no leakage)
- Two validity criteria: C1 (`single_day`) = broader, C2 (`weekly_5of7`) = ~55% smaller/stricter
- Default criterion: C1

### Result Objects

All three `*Results` classes expose:
- `.summary()` → DataFrame
- `.to_csv()` / `.to_json()`
- `.to_submission_yaml()` → paste-ready leaderboard submission body

Leaderboard submissions go via GitHub issue using `.github/ISSUE_TEMPLATE/submission.yml`.

### Lazy Imports

`downstream_evaluation` and `imputation_evaluation` use `__getattr__` in their `__init__.py` to defer heavy imports (sklearn, PyTorch, PyPOTS). Don't break this pattern when adding new public symbols to those packages.
