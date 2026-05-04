"""Dataset download and discovery for OpenMHC.

The dataset is hosted on Harvard Dataverse. This module provides a thin
wrapper around the Dataverse access API so users don't have to remember
the DOI or directory layout.

Two versions are planned:

- ``"tiny"`` — small subset suitable for reviewers and quickstart notebooks
- ``"full"`` — full release matching the paper (not yet published)

Resolution order for the local data directory:

1. Explicit ``data_dir=`` argument
2. ``MHC_DATA_DIR`` environment variable
3. ``~/.cache/openmhc/data`` (default)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_DATAVERSE_BASE = "https://dataverse.harvard.edu"

# DOIs per version. Set to None for versions that haven't been published.
_VERSION_DOIS: dict[str, str | None] = {
    "tiny": "doi:10.7910/DVN/ZYMJF6",
    "full": None,
}

_DEFAULT_CACHE = Path.home() / ".cache" / "openmhc" / "data"


def data_dir(override: str | Path | None = None) -> Path:
    """Resolve the local dataset directory.

    Args:
        override: Explicit path. If provided, returned as-is.

    Returns:
        Absolute path to the dataset directory. May not exist yet — call
        :func:`download_dataset` first.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.getenv("MHC_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CACHE


def download_dataset(
    version: str = "tiny",
    dest: str | Path | None = None,
    api_token: str | None = None,
) -> Path:
    """Download the OpenMHC dataset from Harvard Dataverse.

    Fetches the dataset's ZIP bundle from the Dataverse access API, extracts
    it into the local cache, and returns the path. If the dataset is
    restricted, supply your Dataverse API token via ``api_token=`` or the
    ``DATAVERSE_API_TOKEN`` environment variable.

    Args:
        version: ``"tiny"`` (reviewer subset) or ``"full"`` (paper release).
        dest: Where to put the data. Defaults to :func:`data_dir`.
        api_token: Optional Dataverse API token for restricted datasets.

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        ValueError: If ``version`` is unknown or not yet published.
    """
    if version not in _VERSION_DOIS:
        raise ValueError(
            f"version must be one of {sorted(_VERSION_DOIS)}, got {version!r}"
        )
    doi = _VERSION_DOIS[version]
    if doi is None:
        available = [v for v, d in _VERSION_DOIS.items() if d]
        raise ValueError(
            f"the {version!r} dataset is not yet published. "
            f"Available versions: {available}"
        )

    target = data_dir(dest)
    target.mkdir(parents=True, exist_ok=True)

    token = api_token or os.getenv("DATAVERSE_API_TOKEN")
    headers = {"X-Dataverse-key": token} if token else {}
    url = f"{_DATAVERSE_BASE}/api/access/dataset/:persistentId/?persistentId={doi}"

    print(f"Downloading {version!r} dataset ({doi}) → {target}")
    req = urllib.request.Request(url, headers=headers)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        try:
            with urllib.request.urlopen(req) as resp:
                shutil.copyfileobj(resp, tmp)
        except Exception:
            os.unlink(zip_path)
            raise

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    finally:
        os.unlink(zip_path)

    print(f"Done. Set MHC_DATA_DIR={target} to skip the lookup next time.")
    return target
