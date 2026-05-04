"""Redirect HuggingFace .filter()/.map() cache files away from DVC-tracked dirs.

HF ``Dataset.filter()`` and ``Dataset.map()`` write ``cache-*.arrow`` files into
the dataset directory by default.  This pollutes DVC-tracked directories
(``data/processed/daily_hf/``, ``data/processed/weekly_hf/``) and causes
``dvc pull`` failures.  The helper below generates cache paths under
``data/.hf_cache/`` instead (already gitignored by the ``/data/*`` rule).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / ".hf_cache"


def hf_cache_path(label: str, ds, cache_dir: str | Path | None = None) -> str:
    """Return a cache file path for an HF ``.filter()`` / ``.map()`` call.

    Args:
        label: Short descriptive tag (e.g. ``"daily_filter_train_users"``).
        ds: A ``datasets.Dataset`` instance whose ``_fingerprint`` is used
            for uniqueness.
        cache_dir: Override for the cache directory.  Defaults to
            ``<repo>/data/.hf_cache/``.

    Returns:
        Absolute path string suitable for the ``cache_file_name`` kwarg.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = getattr(ds, "_fingerprint", "nofp")
    token = f"{label}-{fingerprint}"
    short_hash = hashlib.md5(token.encode()).hexdigest()[:12]
    fname = f"{label}-{short_hash}.arrow"
    return str(cache_dir / fname)
