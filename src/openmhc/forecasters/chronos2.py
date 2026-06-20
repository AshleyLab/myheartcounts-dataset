"""Public wrapper for the fine-tuned Chronos-2 forecaster.

The released checkpoint is a full (merged) HuggingFace model directory, so the
internal :class:`~forecasting_evaluation.models.foundational_model.chronos2.Chronos2Model`
loads it directly via ``Chronos2Pipeline.from_pretrained``. Chronos-2 normalizes
internally, so no normalization-stats sidecar is used.

Requires ``pip install 'openmhc[chronos]'`` (the ``chronos`` package).
"""

from __future__ import annotations

from pathlib import Path

from openmhc.forecasters._base import BaseForecaster


class Chronos2Forecaster(BaseForecaster):
    """Released Chronos-2 forecaster (fine-tuned on the MHC training split)."""

    model_name = "chronos2"

    def __init__(
        self,
        model_path: str | Path,
        *,
        normalization_stats_path: str | Path | None = None,
        device: str = "cuda",
        torch_dtype: str = "auto",
    ) -> None:
        """Load the merged Chronos-2 model directory at ``model_path``."""
        # normalization_stats_path is accepted for the from_release contract but
        # unused: Chronos-2 normalizes inputs internally.
        from forecasting_evaluation.config import Chronos2ModelConfig
        from forecasting_evaluation.models.foundational_model.chronos2 import Chronos2Model

        config = Chronos2ModelConfig(
            checkpoint_path=str(model_path),
            device=device,
            torch_dtype=torch_dtype,
        )
        self._model = Chronos2Model(config=config)
