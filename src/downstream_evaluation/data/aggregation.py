"""Aggregation utilities for week-to-user pooling.

Mean-pools week-level features and labels to one row per user.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def aggregate_by_user(
    features: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate week-level data to user-level via pooling.

    Args:
        features: (N_weeks, D) feature array.
        labels: (N_weeks,) label array.
        user_ids: (N_weeks,) user ID array (strings).
        method: Aggregation method. Only "mean" is supported.

    Returns:
        Tuple of (user_features, user_labels, unique_user_ids) where:
        - user_features: (N_users, D) aggregated feature array
        - user_labels: (N_users,) label array (one per user)
        - unique_user_ids: (N_users,) user ID array (strings)
    """
    if method != "mean":
        raise ValueError(f"Unknown aggregation method: {method}. Only 'mean' is supported.")

    # Group features by user
    user_to_features: dict[str, list[np.ndarray]] = defaultdict(list)
    user_to_labels: dict[str, int | float] = {}

    for feat, label, uid in zip(features, labels, user_ids):
        user_to_features[uid].append(feat)
        # Assume consistent label per user (take first occurrence)
        if uid not in user_to_labels:
            user_to_labels[uid] = label

    # Sort users for deterministic ordering
    unique_users = sorted(user_to_features.keys())

    # Mean pool features per user
    agg_features = np.stack([np.mean(user_to_features[uid], axis=0) for uid in unique_users])

    agg_labels = np.array([user_to_labels[uid] for uid in unique_users], dtype=labels.dtype)

    return agg_features, agg_labels, np.array(unique_users, dtype=object)
