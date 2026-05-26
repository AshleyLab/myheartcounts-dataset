"""Dataset download and discovery for OpenMHC.

The dataset is hosted on Harvard Dataverse. This module provides a thin
wrapper around the Dataverse access API so users don't have to remember
the DOI or directory layout.

Two versions are available:

- ``"xs"`` — 593-user subset suitable for reviewers and quickstart notebooks (~1.9 GB)
- ``"full"`` — full 11,894-user release matching the paper (~38 GB, not yet published)

Large benchmark payloads are always resolved from one explicit dataset root:

1. Explicit ``data_dir=`` / ``dest=`` argument
2. ``MHC_DATA_DIR`` environment variable

If neither is provided, OpenMHC raises instead of silently falling back to a
cache directory.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_DATAVERSE_BASE = "https://dataverse.harvard.edu"

# DOIs per version. Set to None for versions that haven't been published yet.
_VERSION_DOIS: dict[str, str | None] = {
    "xs": "doi:10.7910/DVN/ZYMJF6",
    "full": None,  # doi:10.7910/DVN/XNBITM — set once the Dataverse deposit is public
}

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLED_METADATA_DIR = _REPO_ROOT / "data" / "labels"


def bundled_metadata_dir() -> Path:
    """Return the repo-owned directory for small tracked metadata files."""
    return _BUNDLED_METADATA_DIR


def _missing_dataset_root_error() -> ValueError:
    return ValueError(
        "OpenMHC dataset root is required for large benchmark payloads. "
        "Provide it with `data_dir=` / `dest=` or set `MHC_DATA_DIR`. "
        "For example, use `openmhc.download_dataset(dest='~/.cache/openmhc/data')` "
        "or `export MHC_DATA_DIR=~/.cache/openmhc/data`."
    )


def data_dir(override: str | Path | None = None) -> Path:
    """Resolve the explicit dataset directory for large payloads.

    Args:
        override: Explicit path. If provided, returned as-is.

    Returns:
        Absolute path to the dataset directory. May not exist yet.

    Raises:
        ValueError: If neither ``override`` nor ``MHC_DATA_DIR`` is provided.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.getenv("MHC_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    raise _missing_dataset_root_error()


def download_dataset(
    version: str = "xs",
    dest: str | Path | None = None,
    api_token: str | None = None,
) -> Path:
    """Download the OpenMHC dataset from Harvard Dataverse.

    Fetches the dataset's ZIP bundle from the Dataverse access API, extracts
    it into the local cache, and returns the path. If the dataset is
    restricted, supply your Dataverse API token via ``api_token=`` or the
    ``DATAVERSE_API_TOKEN`` environment variable.

    Args:
        version: ``"xs"`` (593-user reviewer subset, ~1.9 GB) or
            ``"full"`` (11,894-user paper release, ~38 GB).
        dest: Where to put the data. Must be provided explicitly here or via
            ``MHC_DATA_DIR``.
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
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        archive_path = tmp.name
        try:
            with urllib.request.urlopen(req) as resp:
                shutil.copyfileobj(resp, tmp)
        except Exception:
            os.unlink(archive_path)
            raise

    try:
        _extract_archive_recursive(Path(archive_path), target)
    finally:
        os.unlink(archive_path)

    if version == "xs":
        _post_process_xs(target)
    elif version == "full":
        _post_process_full(target)

    print(f"Done. Reuse this root via MHC_DATA_DIR={target} or data_dir={target!s}.")
    return target


def _extract_archive_recursive(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, recursively unpacking nested archives.

    Dataverse's dataset-access endpoint always wraps files in an outer zip,
    so a tar.gz upload comes back as zip-of-tarball. After extracting the
    outer archive, any inner ``.tar``, ``.tar.gz``, ``.tgz``, or ``.zip``
    files are unpacked in place and removed.
    """
    _extract_one(archive, dest)
    # Walk dest and unpack any nested archives the outer extract produced.
    for path in list(dest.rglob("*")):
        if not path.is_file():
            continue
        if _looks_like_archive(path):
            _extract_one(path, path.parent)
            path.unlink(missing_ok=True)


def _looks_like_archive(path: Path) -> bool:
    """Detect by suffix; falls through ``.bin`` / ``.zip`` / ``.tar*``."""
    name = path.name.lower()
    return (
        name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar")
        or name.endswith(".zip")
    )


def _extract_one(archive: Path, dest: Path) -> None:
    """Extract a single archive (zip / tar / tar.gz) into ``dest``.

    Detects format via the magic-byte sniffers in :mod:`zipfile` and
    :mod:`tarfile` so the file extension doesn't matter — handy for the
    Dataverse download which we save as ``.bin``.
    """
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
        return
    raise ValueError(
        f"could not detect archive format for {archive!r}; "
        "expected zip or tar/tar.gz"
    )


def _post_process_xs(dest: Path) -> None:
    """Fix directory names and file placement after XS bundle extraction.

    The XS HuggingFace Arrow archives unpack with a version suffix
    (``_tiny`` or ``_xs``) that must be stripped to match the canonical
    paths the evaluation API expects.  Also moves ``normalization_stats.json``
    from the bundle root into ``processed/``.
    """
    processed = dest / "processed"

    # Rename suffixed HF dataset directories to canonical names.
    _rename_suffixed = [
        (processed, "daily_hf"),
        (processed, "daily_hourly_hf"),
        (dest, "hourly_trajectory"),
        (dest, "minute_trajectory"),
    ]
    for parent, canonical in _rename_suffixed:
        target = parent / canonical
        if target.exists():
            continue
        for suffix in ("_xs", "_tiny"):
            candidate = parent / f"{canonical}{suffix}"
            if candidate.exists():
                candidate.rename(target)
                break

    # Move normalization_stats.json from bundle root into processed/.
    root_stats = dest / "normalization_stats.json"
    if root_stats.exists() and not (processed / "normalization_stats.json").exists():
        processed.mkdir(parents=True, exist_ok=True)
        root_stats.rename(processed / "normalization_stats.json")


def _post_process_full(dest: Path) -> None:
    """Handle multi-part archives left in place after full bundle extraction.

    Dataverse serves each ``*.tar.gz.part-NN`` as a separate file.  After the
    outer ZIP is extracted these parts sit in ``archives/`` and must be
    concatenated and streamed into ``tarfile`` before they can be unpacked.
    The part files are removed on success.
    """
    archives_dir = dest / "archives"
    if not archives_dir.exists():
        return

    # Group part files by their base tar.gz name.
    groups: dict[str, list[Path]] = {}
    for part in archives_dir.glob("*.tar.gz.part-*"):
        base = part.name.split(".part-")[0] + ".tar.gz"
        groups.setdefault(base, []).append(part)

    # Resolve each group's extraction target from the archive name.
    _dest_map = {
        "daily_hf_full.tar.gz": dest / "processed",
        "hdf5_sharable_2026_full.tar.gz": dest / "hdf5",
        "minute_trajectory_full.tar.gz": dest,
    }

    for base_name, parts in groups.items():
        parts.sort()
        target_dir = _dest_map.get(base_name, dest)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Stream concatenated parts through tarfile without writing a temp file.
        class _CatStream(io.RawIOBase):
            def __init__(self, paths: list[Path]) -> None:
                self._files = [open(p, "rb") for p in paths]
                self._idx = 0

            def readinto(self, b: bytearray) -> int:
                while self._idx < len(self._files):
                    n = self._files[self._idx].readinto(b)
                    if n:
                        return n
                    self._files[self._idx].close()
                    self._idx += 1
                return 0

        with tarfile.open(fileobj=io.BufferedReader(_CatStream(parts))) as tf:
            tf.extractall(target_dir)

        for part in parts:
            part.unlink(missing_ok=True)
