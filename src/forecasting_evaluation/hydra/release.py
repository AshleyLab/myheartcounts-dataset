"""Release-manifest helpers for forecasting Hydra runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "openmhc_manifest.json"
SPEC_VERSION = 1


@dataclass(frozen=True)
class ForecastingManifest:
    """Parsed forecasting checkpoint manifest."""

    spec_version: int
    kind: str
    arch: dict[str, Any]
    checkpoint_path: Path
    normalization_stats_path: Path | None
    provenance: dict[str, Any]
    manifest_path: Path


def load_forecasting_manifest(path: str | Path) -> ForecastingManifest:
    """Load a forecasting release manifest from a file or release directory."""
    manifest_file = _resolve_manifest_path(path)
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    base = manifest_file.parent

    spec_version = raw.get("spec_version")
    if spec_version != SPEC_VERSION:
        raise ValueError(
            f"Unsupported forecasting manifest spec_version {spec_version!r}; "
            f"expected {SPEC_VERSION}"
        )

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("Forecasting manifest missing required string field 'kind'")

    checkpoint_rel = raw.get("checkpoint")
    if not checkpoint_rel:
        raise ValueError("Forecasting manifest missing required field 'checkpoint'")
    checkpoint_path = (base / checkpoint_rel).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Forecasting manifest references missing checkpoint: {checkpoint_path}"
        )

    stats_rel = raw.get("normalization_stats")
    normalization_stats_path = None
    if stats_rel is not None:
        normalization_stats_path = (base / stats_rel).resolve()
        if not normalization_stats_path.exists():
            raise FileNotFoundError(
                "Forecasting manifest references missing normalization stats: "
                f"{normalization_stats_path}"
            )

    arch = raw.get("arch") or {}
    if not isinstance(arch, dict):
        raise ValueError("Forecasting manifest field 'arch' must be a dict if present")

    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("Forecasting manifest field 'provenance' must be a dict if present")

    return ForecastingManifest(
        spec_version=spec_version,
        kind=kind,
        arch=dict(arch),
        checkpoint_path=checkpoint_path,
        normalization_stats_path=normalization_stats_path,
        provenance=dict(provenance),
        manifest_path=manifest_file,
    )


def _resolve_manifest_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_dir():
        candidate = p / MANIFEST_FILENAME
        if not candidate.exists():
            raise FileNotFoundError(f"No {MANIFEST_FILENAME} in directory {p}")
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {p}")
    return p
