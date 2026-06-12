"""Submission helpers — turn evaluation results into paste-ready YAML.

The output matches the textareas in `.github/ISSUE_TEMPLATE/submission.yml`
so submitters paste once instead of typing per-field.

Skill scores follow the paper convention: for Track 2 imputation, computed
locally from frozen LOCF baselines (`data/baselines/imputation_locf.json`)
using the log-ratio formula. For Track 1, skill score is left as "TBD" —
the maintainer fills it in during ingestion since the Linear baseline per-
task metrics aren't shipped yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from openmhc._results import PredictionResults, ImputationResults

# ---------------------------------------------------------------------------
# Constants from the paper
# ---------------------------------------------------------------------------

# 19-channel sensor layout — categories used for subgroup skill scores
# (see paper Section 4.4 and `scripts/downstream_paper_results/compute_imputation_paper_metrics.py`).
_CHANNEL_CATEGORIES: dict[str, set[str]] = {
    "activity": {"ch_0", "ch_1", "ch_2", "ch_3", "ch_4"},
    "physiology": {"ch_5", "ch_6"},
    "sleep": {"ch_7", "ch_8"},
    "workouts": {f"ch_{i}" for i in range(9, 19)},
}

# Semantic scenarios — binary channels excluded by the paper protocol.
_EXCLUDE_BINARY_SCENARIOS = {"sleep_gap", "workout_gap"}

# Log-ratio clipping bounds (matches paper).
_CLIP_LOWER, _CLIP_UPPER = 1e-3, 1e3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCF_BASELINE_PATH = _REPO_ROOT / "data" / "baselines" / "imputation_locf.json"


# ---------------------------------------------------------------------------
# Skill-score primitive
# ---------------------------------------------------------------------------


def _skill(log_ratios: Iterable[float]) -> float:
    """Convert a set of log error-ratios to a skill score percentage.

    A negative log-ratio means the method beat the baseline, so we negate
    the mean before scaling.
    """
    arr = np.asarray(list(log_ratios), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(-np.mean(arr) * 100.0)


def _channel_category(channel: str) -> str | None:
    """Return the category name for a channel, or None if uncategorised."""
    for cat, channels in _CHANNEL_CATEGORIES.items():
        if channel in channels:
            return cat
    return None


# ---------------------------------------------------------------------------
# Imputation skill scores
# ---------------------------------------------------------------------------


def _load_locf_baseline() -> dict | None:
    """Load the frozen LOCF baseline file, or return None if missing."""
    if not _LOCF_BASELINE_PATH.exists():
        return None
    return json.loads(_LOCF_BASELINE_PATH.read_text())


def _imputation_log_ratios(
    results: "ImputationResults",
    baseline: dict,
) -> list[tuple[str, str, float]]:
    """Compute per-(scenario, channel) log error ratios vs the baseline.

    Channel type is inferred from which metrics the runtime actually
    emitted: ``normalized_rmse`` ⇒ continuous, ``roc_auc`` ⇒ binary.
    Method-side error reads ``normalized_rmse`` / ``roc_auc`` from the
    runtime; baseline-side reads ``nRMSE`` / ``roc_auc`` from the frozen
    LOCF JSON (column names differ because the baseline was extracted
    from the paper-results parquet).
    """
    out: list[tuple[str, str, float]] = []
    bl_scenarios = baseline.get("scenarios", {})
    for scenario, splits in results.scenarios.items():
        bl_channels = bl_scenarios.get(scenario, {})
        test = splits.get("test", {}) if isinstance(splits, dict) else {}
        per_channel = test.get("per_channel") if isinstance(test, dict) else None
        if not isinstance(per_channel, dict):
            continue
        for channel, m in per_channel.items():
            if not isinstance(m, dict):
                continue
            bl_entry = bl_channels.get(channel, {})

            if "normalized_rmse" in m:
                ch_type = "continuous"
            elif "roc_auc" in m:
                ch_type = "binary"
            else:
                continue

            if ch_type == "binary" and scenario in _EXCLUDE_BINARY_SCENARIOS:
                continue

            if ch_type == "continuous":
                e_method = m.get("normalized_rmse")
                e_baseline = bl_entry.get("nRMSE")
            else:  # binary
                auc_method = m.get("roc_auc")
                auc_baseline = bl_entry.get("roc_auc")
                e_method = (1.0 - auc_method) if auc_method is not None else None
                e_baseline = (1.0 - auc_baseline) if auc_baseline is not None else None

            if e_method is None or e_baseline is None:
                continue
            if not (np.isfinite(e_method) and np.isfinite(e_baseline)) or e_baseline <= 0:
                continue
            ratio = float(e_method) / float(e_baseline)
            ratio = float(np.clip(ratio, _CLIP_LOWER, _CLIP_UPPER))
            out.append((scenario, channel, float(np.log(ratio))))
    return out


def _imputation_aggregate_skill(log_ratios: list[tuple[str, str, float]]) -> dict:
    """Aggregate per-channel log ratios into overall + per-category skill scores."""
    if not log_ratios:
        return {
            "skill_score": None,
            "by_category": {k: None for k in _CHANNEL_CATEGORIES},
        }
    overall = _skill(r for _, _, r in log_ratios)
    by_cat: dict[str, float | None] = {}
    for cat in _CHANNEL_CATEGORIES:
        cat_ratios = [r for _, ch, r in log_ratios if _channel_category(ch) == cat]
        by_cat[cat] = _skill(cat_ratios) if cat_ratios else None
    return {"skill_score": overall, "by_category": by_cat}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def imputation_to_submission_yaml(
    results: "ImputationResults",
    method_name: str,
    submitter_team: str,
    code_url: str,
    paper_url: str = "",
    method_category: str = "Other",
    foundation_variant: str = "N/A (not a foundation model)",
    feature_dim: str = "—",
    notes: str = "",
) -> str:
    """Render a paste-ready submission body for a Track 2 imputation result.

    Args:
        results: ImputationResults from `evaluate_imputation`.
        method_name: Short, citation-ready name (e.g. "MeanImputer").
        submitter_team: Lab / company affiliation for attribution.
        code_url: Public repo URL (required — needed for reproducibility).
        paper_url: Paper / preprint / blog / slides URL. Optional; leave
            empty for independent teams without a write-up.
        method_category: One of the dropdown options in submission.yml.
        foundation_variant: One of the dropdown options.
        feature_dim: Latent / embedding dim, or "—".
        notes: Free-form notes for reviewers.

    Returns:
        Plain-text body matching the textareas in submission.yml. Paste into
        the "Aggregate metrics", "Subgroup skill scores", and "Raw per-sub-task
        metrics" fields, then fill the simple inputs (method name, etc.) at
        the top of the form.
    """
    baseline = _load_locf_baseline()
    if baseline is not None:
        log_ratios = _imputation_log_ratios(results, baseline)
        agg = _imputation_aggregate_skill(log_ratios)
    else:
        agg = {
            "skill_score": None,
            "by_category": {k: None for k in _CHANNEL_CATEGORIES},
        }

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None and np.isfinite(v) else "—"

    raw = json.dumps(results.scenarios, indent=2, default=_json_default)

    return _render_yaml_block(
        track="Track 2 — Imputation (Daily, single-day context)",
        method_name=method_name,
        submitter_team=submitter_team,
        method_category=method_category,
        foundation_variant=foundation_variant,
        feature_dim=feature_dim,
        paper_url=paper_url,
        code_url=code_url,
        notes=notes,
        aggregate_lines=[
            f"skill_score: {_fmt(agg['skill_score'])}",
            "fair_skill_score: —  # computed by maintainers from subgroup runs",
            "avg_rank: —  # computed by maintainers vs current leaderboard",
        ],
        subgroup_lines=[
            f"activity: {_fmt(agg['by_category'].get('activity'))}",
            f"physiology: {_fmt(agg['by_category'].get('physiology'))}",
            f"sleep: {_fmt(agg['by_category'].get('sleep'))}",
            f"workouts: {_fmt(agg['by_category'].get('workouts'))}",
            "semantic: —  # paper-only category; ignore unless reporting it explicitly",
        ],
        raw_block=raw,
    )


def prediction_to_submission_yaml(
    results: "PredictionResults",
    method_name: str,
    submitter_team: str,
    code_url: str,
    paper_url: str = "",
    track: str = "Track 1 — Health & Behavior Outcome Prediction (Static)",
    method_category: str = "Other",
    foundation_variant: str = "N/A (not a foundation model)",
    feature_dim: str = "—",
    notes: str = "",
) -> str:
    """Render a paste-ready submission body for a Track 1 outcome-prediction result.

    Aggregate skill score and subgroup skill scores are emitted as "—"
    placeholders for now — the public repo doesn't yet ship the Linear-
    baseline per-task metrics needed to compute them. The maintainers fill
    these in from raw_metrics during ingestion.

    Args, Returns: see :func:`imputation_to_submission_yaml`.
    """
    raw = json.dumps(results.records, indent=2, default=_json_default)
    return _render_yaml_block(
        track=track,
        method_name=method_name,
        submitter_team=submitter_team,
        method_category=method_category,
        foundation_variant=foundation_variant,
        feature_dim=feature_dim,
        paper_url=paper_url,
        code_url=code_url,
        notes=notes,
        aggregate_lines=[
            "skill_score: —  # computed by maintainers from raw_metrics",
            "fair_skill_score: —",
            "avg_rank: —",
        ],
        subgroup_lines=[
            "demographics: —  # computed by maintainers",
            "medical_conditions: —",
            "body_biomarkers: —",
            "mental_wellbeing: —",
            "sleep_lifestyle: —",
        ],
        raw_block=raw,
    )


def forecasting_to_submission_yaml(
    results: "ForecastingResults",
    method_name: str,
    submitter_team: str,
    code_url: str,
    paper_url: str = "",
    method_category: str = "Other",
    foundation_variant: str = "N/A (not a foundation model)",
    feature_dim: str = "—",
    notes: str = "",
) -> str:
    """Render a paste-ready submission body for a Track 3 forecasting result.

    Skill scores against the Seasonal Naive baseline are left as ``—``
    until the per-channel baseline file is shipped. Subgroup keys match
    the submission template (``step``, ``flights``, ``hr``, ``energy``,
    ``sleep``, ``workouts``).
    """
    raw = json.dumps(results.per_channel, indent=2, default=_json_default)
    return _render_yaml_block(
        track="Track 3 — Forecasting",
        method_name=method_name,
        submitter_team=submitter_team,
        method_category=method_category,
        foundation_variant=foundation_variant,
        feature_dim=feature_dim,
        paper_url=paper_url,
        code_url=code_url,
        notes=notes,
        aggregate_lines=[
            "skill_score: —  # computed by maintainers vs Seasonal Naive baseline",
            "fair_skill_score: —",
            "avg_rank: —",
        ],
        subgroup_lines=[
            "step: —",
            "flights: —",
            "hr: —",
            "energy: —",
            "sleep: —",
            "workouts: —",
        ],
        raw_block=raw,
    )


# ---------------------------------------------------------------------------
# Internal rendering
# ---------------------------------------------------------------------------


def _render_yaml_block(
    *,
    track: str,
    method_name: str,
    submitter_team: str,
    method_category: str,
    foundation_variant: str,
    feature_dim: str,
    paper_url: str,
    code_url: str,
    notes: str,
    aggregate_lines: list[str],
    subgroup_lines: list[str],
    raw_block: str,
) -> str:
    """Format a complete paste-ready submission body."""
    indent = "  "
    parts = [
        f"# Paste-ready submission body for: {method_name}",
        f"# Track: {track}",
        "#",
        "# Single-input fields (top of the form):",
        f"#   method_name:       {method_name}",
        f"#   submitter_team:    {submitter_team}",
        f"#   track:             {track}",
        f"#   method_category:   {method_category}",
        f"#   foundation_variant: {foundation_variant}",
        f"#   feature_dim:       {feature_dim}",
        f"#   paper_url:         {paper_url or '(leave blank — no paper)'}",
        f"#   code_url:          {code_url}",
        "",
        "# --- aggregate_metrics (textarea) ---",
        *aggregate_lines,
        "",
        "# --- subgroup_metrics (textarea) ---",
        *subgroup_lines,
        "",
        "# --- raw_metrics (optional textarea) ---",
        raw_block,
    ]
    if notes:
        parts.extend(["", "# --- notes ---", notes])
    return "\n".join(parts)


def _json_default(obj):
    """JSON serializer for numpy scalars / arrays inside results."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
