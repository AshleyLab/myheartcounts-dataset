"""Manifest format for distributing imputer checkpoints.

A "release" is a directory with three files::

    my-release/
    ├── model.{pypots,ckpt,...}
    ├── normalization_stats.json
    └── openmhc_manifest.json

The manifest captures the training-time invariants — which model class,
which architecture hyperparameters, which stats file — so loading at
inference time is a single call::

    imp = BRITSImputer.from_release("my-release/")

Paths inside the manifest are stored *relative to the manifest file* so
the whole directory is movable. ``normalization_stats`` may be ``null``
for checkpoints trained on raw (un-normalized) inputs.

The :class:`ReleaseLoadableMixin` is shared across imputer families
(PyPOTS, LSM2, ...) — subclasses set ``model_name`` to match the
manifest's ``kind`` field and accept ``manifest.arch`` as constructor
kwargs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "openmhc_manifest.json"
SPEC_VERSION = 1

_KNOWN_KINDS = frozenset({
    "brits", "timesnet", "dlinear", "fedformer",
    "lsm2", "lsm2_weekly_sparse",
})


@dataclass(frozen=True)
class Manifest:
    """Parsed, path-resolved release manifest.

    ``checkpoint_path`` and ``normalization_stats_path`` are absolute
    paths resolved against the manifest file's directory. ``arch`` is
    the dict of training-time architecture kwargs, ready to splat into
    the wrapper's constructor.
    """

    spec_version: int
    kind: str
    arch: dict[str, Any]
    checkpoint_path: Path
    normalization_stats_path: Path | None
    provenance: dict[str, Any]
    manifest_path: Path


def _resolve_manifest_path(path: str | Path) -> Path:
    """Accept either the manifest file itself or a directory containing it."""
    p = Path(path)
    if p.is_dir():
        candidate = p / MANIFEST_FILENAME
        if not candidate.exists():
            raise FileNotFoundError(
                f"No {MANIFEST_FILENAME} in directory {p}"
            )
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {p}")
    return p


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate an openmhc release manifest.

    Args:
        path: Either a manifest file (``openmhc_manifest.json``) or a
            directory containing one.

    Returns:
        A :class:`Manifest` with ``checkpoint_path`` and
        ``normalization_stats_path`` resolved against the manifest's
        directory.

    Raises:
        FileNotFoundError: If the manifest, checkpoint, or stats file is
            missing.
        ValueError: If the manifest schema is invalid.
    """
    manifest_file = _resolve_manifest_path(path)
    raw = json.loads(manifest_file.read_text())
    base = manifest_file.parent

    spec_version = raw.get("spec_version")
    if spec_version != SPEC_VERSION:
        raise ValueError(
            f"Unsupported manifest spec_version {spec_version!r}; "
            f"this build understands spec_version={SPEC_VERSION}"
        )

    kind = raw.get("kind")
    if kind not in _KNOWN_KINDS:
        raise ValueError(
            f"Unknown manifest kind {kind!r}; expected one of {sorted(_KNOWN_KINDS)}"
        )

    checkpoint_rel = raw.get("checkpoint")
    if not checkpoint_rel:
        raise ValueError("Manifest missing required field 'checkpoint'")
    checkpoint_path = (base / checkpoint_rel).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Manifest references missing checkpoint: {checkpoint_path}"
        )

    stats_rel = raw.get("normalization_stats")
    if stats_rel is None:
        stats_path: Path | None = None
    else:
        stats_path = (base / stats_rel).resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Manifest references missing stats file: {stats_path}"
            )

    arch = raw.get("arch")
    if not isinstance(arch, dict):
        raise ValueError("Manifest field 'arch' must be a dict")

    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("Manifest field 'provenance' must be a dict if present")

    return Manifest(
        spec_version=spec_version,
        kind=kind,
        arch=dict(arch),
        checkpoint_path=checkpoint_path,
        normalization_stats_path=stats_path,
        provenance=dict(provenance),
        manifest_path=manifest_file,
    )


def write_manifest(
    directory: str | Path,
    *,
    kind: str,
    arch: dict[str, Any],
    checkpoint: str,
    normalization_stats: str | None = None,
    provenance: dict[str, Any] | None = None,
    filename: str = MANIFEST_FILENAME,
) -> Path:
    """Write a release manifest into ``directory``.

    Paths in ``checkpoint`` and ``normalization_stats`` are stored as-is
    and interpreted at load time relative to the manifest's directory —
    typically just filenames pointing at siblings.

    Args:
        directory: Release directory; will be created if missing.
        kind: Model kind, one of ``{"brits", "timesnet", "dlinear", "fedformer"}``.
        arch: Training-time architecture kwargs (e.g. ``{"n_steps": 1440,
            "n_features": 19, "rnn_hidden_size": 128}`` for BRITS).
        checkpoint: Path to the ``.pypots`` file, relative to ``directory``.
        normalization_stats: Path to the stats JSON, relative to
            ``directory``. ``None`` if the model expects raw inputs.
        provenance: Optional metadata (training run id, dataset version,
            paper table, etc.) — not interpreted, just stored.
        filename: Manifest filename (defaults to ``openmhc_manifest.json``).

    Returns:
        Path to the written manifest file.
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(
            f"Unknown manifest kind {kind!r}; expected one of {sorted(_KNOWN_KINDS)}"
        )
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": SPEC_VERSION,
        "kind": kind,
        "checkpoint": checkpoint,
        "normalization_stats": normalization_stats,
        "arch": dict(arch),
        "provenance": dict(provenance) if provenance else {},
    }
    out = out_dir / filename
    out.write_text(json.dumps(payload, indent=2))
    return out


class ReleaseLoadableMixin:
    """Adds :meth:`from_release` to any imputer wrapper class.

    Subclasses must set the class-level attribute ``model_name`` to a value
    that matches the manifest's ``kind`` field (e.g. ``"brits"``, ``"lsm2"``).
    The constructor must accept ``model_path``, ``normalization_stats_path``,
    and every key in the manifest's ``arch`` dict as keyword arguments.
    """

    model_name: str = ""

    @classmethod
    def from_release(cls, path: str | Path, **runtime_kwargs):
        """Construct from a release directory containing an ``openmhc_manifest.json``.

        The manifest captures training-time invariants (model kind,
        architecture hyperparameters, paths to the checkpoint and
        normalization stats). Runtime knobs like ``device``,
        ``inference_batch_size``, and ``data_dir`` can be passed as
        keyword arguments.

        Args:
            path: A release directory or a direct path to a manifest file.
            **runtime_kwargs: Forwarded to the constructor (e.g.
                ``device="cuda:0"``). Must not duplicate any key in the
                manifest's ``arch`` dict.

        Returns:
            An instance of the calling class.

        Raises:
            ValueError: If the manifest's ``kind`` does not match
                ``cls.model_name``.
        """
        manifest = load_manifest(path)
        if manifest.kind != cls.model_name:
            raise ValueError(
                f"Manifest is for kind {manifest.kind!r}, but "
                f"{cls.__name__} expects kind {cls.model_name!r}. "
                f"Use the matching wrapper class."
            )
        stats_path = (
            str(manifest.normalization_stats_path)
            if manifest.normalization_stats_path is not None
            else None
        )
        return cls(
            model_path=str(manifest.checkpoint_path),
            normalization_stats_path=stats_path,
            **manifest.arch,
            **runtime_kwargs,
        )
