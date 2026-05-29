"""Discoverable accessor for MHC context labels.

Context labels (e.g. ``field_smokingHistory``, ``field_race``, ``field_device_iphone``)
are participant-level covariates -- distinct from the prediction targets in
``last_labels.json``.  They are queryable via :func:`labels.get_labels` already;
this module adds a typed entry point that validates the label is in fact a
context variable, preventing accidental queries on prediction targets.

Quick start:

    >>> from context import get_context, CONTEXT_NAMES
    >>> import pandas as pd
    >>> result = get_context(
    ...     health_code="user-123",
    ...     timestamp=pd.Timestamp("2020-01-01"),
    ...     context_label="field_smokingHistory",
    ... )
    >>> result.value
    False
"""

from __future__ import annotations

import pandas as pd

from labels.api import CONTEXT_NAMES, LABEL_NAMES, LabelResult, get_labels


def get_context(
    health_code: str,
    timestamp: pd.Timestamp,
    context_label: str,
    enforce_type: bool = True,
    return_valid_only: bool = True,
) -> LabelResult:
    """Return the nearest-in-time context-label value for a healthCode.

    Identical semantics to :func:`labels.get_labels`, but validates that the
    label is in :data:`CONTEXT_NAMES` so targets cannot be queried by mistake.

    Args:
        health_code: The participant's health code.
        timestamp: Reference timestamp for nearest-match lookup.
        context_label: A label name from :data:`CONTEXT_NAMES`.
        enforce_type: Forwarded to :func:`labels.get_labels`.
        return_valid_only: Forwarded to :func:`labels.get_labels`.

    Returns:
        LabelResult with the matched timestamp and value.

    Raises:
        ValueError: If ``context_label`` is not in :data:`CONTEXT_NAMES`.
        KeyError: If the healthCode is not found for this label.
    """
    if context_label not in LABEL_NAMES:
        raise ValueError(f"Unknown label: {context_label}")
    if context_label not in CONTEXT_NAMES:
        raise ValueError(
            f"'{context_label}' is not a context label "
            f"(it is a prediction target). Use labels.get_labels() instead."
        )
    return get_labels(
        health_code,
        timestamp,
        context_label,
        enforce_type=enforce_type,
        return_valid_only=return_valid_only,
    )
