"""Public WBM encoder wrapper (Track 1 — outcome prediction).

The reported **WBM** model is the WBM-encoder-primary + Linear-fallback hybrid
(see :mod:`downstream_evaluation.models.hybrid_wbm`). This module exposes it as a
release-loadable :class:`~openmhc.Method` so a fresh user can fetch the published
checkpoint and reproduce the leaderboard entry with a single call::

    >>> import openmhc
    >>> from openmhc.encoders import WBM
    >>> enc = WBM.from_release("hf://MyHeartCounts/openmhc-wbm-dp")
    >>> results = openmhc.evaluate_prediction(enc, version="full")

``from_release`` resolves the bundle's ``model.ckpt`` (the Mamba2 contrastive
encoder weights) and ``normalization_stats.json``, then constructs the hybrid.
The Mamba2 kernels (``mamba_ssm``) are CUDA-only, so running the encoder requires
a GPU; loading/validating the bundle's manifest does not.

The published bundle lives at ``MyHeartCounts/openmhc-wbm-dp`` on the Hugging
Face Hub.
"""

from __future__ import annotations

import numpy as np

from openmhc.encoders._release import Manifest, ReleaseLoadableMixin, load_manifest, write_manifest

__all__ = ["WBM", "Manifest", "load_manifest", "write_manifest"]


class WBM(ReleaseLoadableMixin):
    """Release-loadable wrapper for the reported WBM model (hybrid, full cohort).

    Implements the public :class:`~openmhc.Method` contract by delegating to
    :class:`downstream_evaluation.models.hybrid_wbm.Hybrid`. The class attributes
    mirror ``Hybrid`` so the evaluation engine routes the daily cohort and pools
    raw daily segments for the Linear fallback branch.
    """

    model_name = "wbm"
    # Mirror downstream_evaluation.models.hybrid_wbm.Hybrid so the engine drives
    # this wrapper identically (full daily cohort + Linear-fallback segments).
    name = "wbm"
    input_granularity = "daily"
    needs_segments = True

    def __init__(
        self,
        model_path: str,
        *,
        normalization_stats_path: str | None = None,
        data_dir: str | None = None,
        seed: int = 42,
        in_dim: int = 38,
        embed_dim: int = 256,
        hidden_dim: int = 64,
        num_layers: int = 4,
        proj_dim: int = 128,
        dropout: float = 0.223,
    ) -> None:
        """Build the hybrid from a released checkpoint.

        Args:
            model_path: Path (or resolved ``hf://``/``wandb:`` ref) to the
                Lightning ``.ckpt`` encoder checkpoint.
            normalization_stats_path: Path to the bundle's ``normalization_stats.json``.
                When ``None``, the encoder falls back to the dataset's canonical
                ``normalization_stats_hourly.json``.
            data_dir: Dataset root (else ``MHC_DATA_DIR`` / default cache).
            seed: RNG seed for the probe / PCA.
            in_dim: Encoder input channels (manifest ``arch``).
            embed_dim: Representation dimensionality (manifest ``arch``).
            hidden_dim: Mamba2 hidden width (manifest ``arch``).
            num_layers: Number of Mamba2 layers (manifest ``arch``).
            proj_dim: Projection-head dimensionality (manifest ``arch``).
            dropout: Dropout rate (manifest ``arch``).

        The architecture dims are validated against the checkpoint's known
        architecture (fail loudly on drift).
        """
        self._validate_arch(
            dict(
                in_dim=in_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                proj_dim=proj_dim,
                dropout=dropout,
            )
        )
        # Lazy import keeps `from openmhc.encoders import WBM` light — the heavy
        # downstream_evaluation engine is pulled only when a model is built.
        from downstream_evaluation.models.hybrid_wbm import Hybrid

        self._hybrid = Hybrid(
            data_dir=data_dir,
            checkpoint=str(model_path),
            seed=seed,
            normalization_stats_path=normalization_stats_path,
        )

    @staticmethod
    def _validate_arch(arch: dict) -> None:
        """Assert the manifest's arch matches the checkpoint's known architecture."""
        from downstream_evaluation.models.wbm.model import _ARCH

        if arch != dict(_ARCH):
            raise ValueError(
                f"Manifest arch {arch} does not match the WBM checkpoint architecture "
                f"{dict(_ARCH)}. The published checkpoint and wrapper are out of sync."
            )

    # -- Method protocol: delegate to the hybrid ----------------------------- #
    def set_context(self, ctx) -> None:
        """Forward the per-(task, split) cohort context to the hybrid."""
        self._hybrid.set_context(ctx)

    def set_loader(self, loader) -> None:
        """Forward the shared DataLoader to the hybrid's SSL branch."""
        self._hybrid.set_loader(loader)

    def fit(self, data, labels, task_type) -> None:
        """Fit both branches (SSL probe + Linear fallback)."""
        self._hybrid.fit(data, labels, task_type)

    def predict(self, data) -> np.ndarray:
        """Per-user routed predictions, aligned with the cohort order."""
        return self._hybrid.predict(data)
