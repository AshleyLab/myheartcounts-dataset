"""User split utilities for imputation evaluation.

Copied from downstream_evaluation/data/splits.py for independence. 
# FIXME: Copying code is not a good idea -> manual review this and ensure copying is the correct approach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_split_file(path: Path) -> dict[str, set[str]]:
    """Load user split mapping from JSON file."""
    data = json.loads(path.read_text())
    return {k: set(v) for k, v in data.items()}


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
