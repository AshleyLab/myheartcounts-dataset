"""User split utilities for downstream evaluation.

Standalone split helpers (no pytorch_lightning dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REQUIRED_SPLIT_KEYS = {"train", "validation", "test"}


def load_split_file(path: Path) -> dict[str, set[str]]:
    """Load user split mapping from JSON file.

    Validates that the file contains the required keys ("train",
    "validation", "test") and that each value is a list of strings.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    data = json.loads(path.read_text())

    missing_keys = _REQUIRED_SPLIT_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Split file {path} missing required keys: {missing_keys}. "
            f"Found keys: {sorted(data.keys())}. "
            f"Expected: {sorted(_REQUIRED_SPLIT_KEYS)}"
        )

    splits = {k: set(v) for k, v in data.items() if k in _REQUIRED_SPLIT_KEYS}

    # Check for user overlap between splits
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = splits[a] & splits[b]
        if overlap:
            raise ValueError(
                f"Split file {path} has {len(overlap)} users in both "
                f"'{a}' and '{b}': {list(overlap)[:5]}..."
            )

    return splits


def random_split_users(
    user_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    """Create a deterministic user-based split."""
    rng = np.random.default_rng(seed)
    shuffled = list(user_ids)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = int(n_total * val_ratio)
    train_users = set(shuffled[:n_train])
    val_users = set(shuffled[n_train : n_train + n_val])
    test_users = set(shuffled[n_train + n_val :])
    return {"train": train_users, "validation": val_users, "test": test_users}
