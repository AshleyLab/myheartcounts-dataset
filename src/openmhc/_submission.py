"""Submission helpers — render the leaderboard submission packet.

A submission is a pull request on the Hugging Face dataset repo
``MyHeartCounts/OpenMHC-leaderboard-data`` that adds two files under the
track's subdirectory:

- ``<track>/<method>.parquet``   — the per-user substrate produced by the eval
- ``<track>/<method>.meta.json`` — the display sidecar the leaderboard reads
  (``display_name``, ``type``, ``submitter``, ``subtrack``)

The maintainers recompute the paired skill score, fair skill score, and average
rank from the substrate during ingestion (LOCF baseline for Track 2, Linear for
Track 1, Seasonal Naive for Track 3). The reducer is the same paired
user-bootstrap formula implemented in
``imputation_evaluation/evaluation/bootstrap_skill_rank.py`` (Track 2) and
``forecasting_evaluation/metrics/skill_score_summary.py`` (Track 3).

These helpers render the ``meta.json`` block plus the PR file checklist so
submitters fill it once instead of hand-writing the sidecar. The substrate
parquet itself is produced by the evaluation run, not by these helpers.
"""

from __future__ import annotations

import json
import re

LEADERBOARD_REPO = "MyHeartCounts/OpenMHC-leaderboard-data"
LEADERBOARD_REPO_URL = f"https://huggingface.co/datasets/{LEADERBOARD_REPO}"

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def imputation_to_submission_yaml(
    method_name: str,
    submitter_team: str,
    code_url: str,
    paper_url: str = "",
    method_category: str = "Other",
    foundation_variant: str = "N/A (not a foundation model)",
    feature_dim: str = "—",
    notes: str = "",
) -> str:
    """Render the Track 2 (imputation) leaderboard submission packet.

    Produces the ``<method>.meta.json`` sidecar and the pull-request file
    checklist for ``MyHeartCounts/OpenMHC-leaderboard-data``. The per-user
    substrate parquet itself comes from the evaluation run — pass ``output_dir=``
    to ``evaluate_imputation`` to write ``per_user_errors.parquet`` — and the
    maintainers compute the paired skill / fair-skill / rank from it vs. LOCF.

    Args:
        method_name: Short, citation-ready name (e.g. "MeanImputer"). Used as
            the sidecar ``display_name``.
        submitter_team: Lab / company affiliation; the sidecar ``submitter``.
        code_url: Public repo URL — recorded in the PR description for
            reproducibility.
        paper_url: Paper / preprint / blog / slides URL. Optional; leave empty
            for independent teams without a write-up.
        method_category: The sidecar ``type`` (e.g. "Statistical / Classical
            baseline").
        foundation_variant: Foundation-model variant, for the PR description.
        feature_dim: Latent / embedding dim, or "—".
        notes: Free-form notes for reviewers.

    Returns:
        Plain-text packet: a copy-paste ``meta.json`` block plus the PR
        instructions for the imputation track.
    """
    return _render_packet(
        track="Track 2 — Imputation (Daily, single-day context)",
        track_subdir="imputation",
        subtrack="single-day",
        subtrack_options="single-day | long-context",
        substrate_hint="per-user substrate (e.g. per_user_errors.parquet from output_dir=)",
        subdir_finalized=True,
        method_name=method_name,
        method_category=method_category,
        submitter_team=submitter_team,
        code_url=code_url,
        paper_url=paper_url,
        feature_dim=feature_dim,
        foundation_variant=foundation_variant,
        notes=notes,
    )


def prediction_to_submission_yaml(
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
    """Render the Track 1 (outcome prediction) leaderboard submission packet.

    Same shape as :func:`imputation_to_submission_yaml`. The Track 1 subdir
    name and substrate format are still being finalized, so the rendered packet
    carries a NOTE flagging that; maintainers compute skill / fair-skill / rank
    vs. the Linear baseline during ingestion.

    Args, Returns: see :func:`imputation_to_submission_yaml`. ``track`` selects
    the Static vs. Longitudinal sub-track label.
    """
    return _render_packet(
        track=track,
        track_subdir="prediction",
        subtrack="other",
        subtrack_options="finalized in the per-track update",
        substrate_hint="per-user substrate from the eval (format being finalized)",
        subdir_finalized=False,
        method_name=method_name,
        method_category=method_category,
        submitter_team=submitter_team,
        code_url=code_url,
        paper_url=paper_url,
        feature_dim=feature_dim,
        foundation_variant=foundation_variant,
        notes=notes,
    )


def forecasting_to_submission_yaml(
    method_name: str,
    submitter_team: str,
    code_url: str,
    paper_url: str = "",
    method_category: str = "Other",
    foundation_variant: str = "N/A (not a foundation model)",
    feature_dim: str = "—",
    notes: str = "",
) -> str:
    """Render the Track 3 (forecasting) leaderboard submission packet.

    Same shape as :func:`imputation_to_submission_yaml`. The Track 3 subdir
    name and substrate format are still being finalized, so the rendered packet
    carries a NOTE flagging that; maintainers compute skill / fair-skill / rank
    vs. the Seasonal Naive baseline during ingestion.

    Args, Returns: see :func:`imputation_to_submission_yaml`.
    """
    return _render_packet(
        track="Track 3 — Forecasting",
        track_subdir="forecasting",
        subtrack="other",
        subtrack_options="finalized in the per-track update",
        substrate_hint="per-user substrate from the eval (format being finalized)",
        subdir_finalized=False,
        method_name=method_name,
        method_category=method_category,
        submitter_team=submitter_team,
        code_url=code_url,
        paper_url=paper_url,
        feature_dim=feature_dim,
        foundation_variant=foundation_variant,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Internal rendering
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    """Lowercase filename stem suggestion derived from a method name."""
    s = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
    return s or "method"


def _render_packet(
    *,
    track: str,
    track_subdir: str,
    subtrack: str,
    subtrack_options: str,
    substrate_hint: str,
    subdir_finalized: bool,
    method_name: str,
    method_category: str,
    submitter_team: str,
    code_url: str,
    paper_url: str,
    feature_dim: str,
    foundation_variant: str,
    notes: str,
) -> str:
    """Format the meta.json sidecar block plus the PR file checklist."""
    method = _slug(method_name)
    meta = {
        "display_name": method_name,
        "type": method_category,
        "submitter": submitter_team,
        "subtrack": subtrack,
    }
    parts = [
        f"# Leaderboard submission packet — {method_name}",
        f"# Track: {track}",
        "#",
        "# Submit by opening a pull request on the leaderboard dataset:",
        f"#   {LEADERBOARD_REPO_URL}",
        "#",
        f"# Add two files under {track_subdir}/ (pick your own <method> stem):",
        f"#   {track_subdir}/{method}.parquet     # {substrate_hint}",
        f"#   {track_subdir}/{method}.meta.json   # the block below",
    ]
    if not subdir_finalized:
        parts += [
            "#",
            f"# NOTE: the {track_subdir}/ subdir name and substrate format for this",
            "# track are being finalized — confirm against tools/leaderboard_docs/",
            "# and tools/upload_leaderboard_substrate.py before submitting.",
        ]
    parts += [
        "",
        f"# --- {method}.meta.json ---",
        json.dumps(meta, indent=2),
        f"# subtrack options: {subtrack_options}",
        "",
        "# --- for the PR description (not uploaded) ---",
        f"#   code_url:           {code_url}",
        f"#   paper_url:          {paper_url or '(none)'}",
        f"#   feature_dim:        {feature_dim}",
        f"#   foundation_variant: {foundation_variant}",
    ]
    if notes:
        parts += ["", "# --- notes ---", notes]
    return "\n".join(parts)
