"""Submission helpers — turn evaluation results into paste-ready YAML.

The output matches the textareas in `.github/ISSUE_TEMPLATE/submission.yml`
so submitters paste once instead of typing per-field.

Skill scores follow the same maintainer-filled convention across all three
tracks: submitters paste absolute per-channel metrics (MAE / AUC) and the
maintainers compute the paired skill score, fair skill score, and average
rank against the per-track baseline (LOCF for Track 2, Linear for Track 1,
Seasonal Naive for Track 3) during ingestion. This keeps the estimand
consistent: it's always the maintainer-side paired user-bootstrap formula
implemented in ``imputation_evaluation/evaluation/bootstrap_skill_rank.py``
(Track 2) and ``forecasting_evaluation/metrics/skill_score_summary.py``
(Track 3).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmhc._results import (
        ForecastingResults,
        ImputationResults,
        PredictionResults,
    )

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def imputation_to_submission_yaml(
    results: ImputationResults,
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

    Skill score, fair skill score, average rank, and per-category subgroup
    scores are emitted as ``—`` placeholders. The maintainers fill them in
    from the absolute per-channel metrics in the ``raw_metrics`` block,
    running the paired user-bootstrap reducer in
    ``imputation_evaluation.evaluation.bootstrap_skill_rank``
    (paired geomean of clipped per-user ratios, MAE for continuous,
    ``max(1 − AUC_u, 0.005)`` for binary — parity with Track 3 forecasting).

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
            "skill_score: —  # computed by maintainers vs LOCF baseline",
            "fair_skill_score: —  # computed by maintainers from the disparity-ratio bootstrap",
            "avg_rank: —  # computed by maintainers vs current leaderboard",
        ],
        subgroup_lines=[
            "activity: —  # computed by maintainers",
            "physiology: —",
            "sleep: —",
            "workouts: —",
            "semantic: —  # paper-only category; ignore unless reporting it explicitly",
        ],
        raw_block=raw,
    )


def prediction_to_submission_yaml(
    results: PredictionResults,
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
    results: ForecastingResults,
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
