"""Public wrapper for the fine-tuned Toto forecaster.

The released checkpoint is a Lightning ``.ckpt``; the internal
:class:`~forecasting_evaluation.models.foundational_model.toto.TotoModel`
loads it onto the ``Datadog/Toto-Open-Base-1.0`` backbone (merging LoRA deltas
when present). Toto normalizes internally, so no normalization-stats sidecar
is used.

Requires ``pip install 'openmhc[toto]'`` (the ``toto-ts`` package).
"""

from __future__ import annotations

from pathlib import Path

from openmhc.forecasters._base import BaseForecaster


class TotoForecaster(BaseForecaster):
    """Released Toto forecaster (fine-tuned on the MHC training split)."""

    model_name = "toto"

    def __init__(
        self,
        model_path: str | Path,
        *,
        normalization_stats_path: str | Path | None = None,
        device: str = "cuda",
        lora_alpha: float | None = None,
        context_length: int = 2048,
        num_samples: int = 256,
    ) -> None:
        """Load the Toto ``.ckpt`` at ``model_path`` onto the Toto backbone."""
        # normalization_stats_path is accepted for the from_release contract but
        # unused: Toto normalizes inputs internally.
        from forecasting_evaluation.models.foundational_model.toto import (
            TotoModel,
            TotoModelConfig,
        )

        config = TotoModelConfig(
            checkpoint_path=str(model_path),
            device=device,
            lora_alpha=lora_alpha,
            context_length=int(context_length),
            num_samples=int(num_samples),
        )
        self._model = TotoModel(config=config)
