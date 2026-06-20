"""Per-run-dir artifact writer.

Defines the on-disk layout each Hydra-driven evaluation run produces, so all
three tracks land artifacts in the same place under their respective
``hydra.run.dir`` / ``hydra.sweep.subdir``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from openmhc.imputers._release import Manifest


def write_run_artifacts(
    run_dir: Path,
    *,
    resolved_cfg: DictConfig,
    manifest: Manifest | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """Write the standard per-run artifact set into ``run_dir``.

    Produces:

    - ``resolved_config.yaml``: full post-interpolation config snapshot.
    - ``openmhc_manifest.json``: copied from the loaded checkpoint release when
      ``manifest`` is provided (imputation paper-checkpoint runs only today).
    - ``wandb_run_id.txt``: back-link to W&B when applicable.

    ``.hydra/overrides.yaml`` is written by Hydra itself; we don't touch it.

    Args:
        run_dir: Hydra-managed run directory (typically
            ``Path(HydraConfig.get().runtime.output_dir)``).
        resolved_cfg: The fully-resolved DictConfig to snapshot.
        manifest: Optional checkpoint-release manifest. Its
            ``manifest_path`` is copied into ``run_dir``.
        wandb_run_id: Optional wandb run ID to record.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=resolved_cfg, f=run_dir / "resolved_config.yaml")

    if manifest is not None and manifest.manifest_path is not None:
        src = Path(manifest.manifest_path)
        if src.exists():
            shutil.copy(src, run_dir / "openmhc_manifest.json")

    if wandb_run_id:
        (run_dir / "wandb_run_id.txt").write_text(wandb_run_id + "\n")
