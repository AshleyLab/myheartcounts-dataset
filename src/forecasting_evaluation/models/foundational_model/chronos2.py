"""Chronos-2 foundational forecasting model wrapper."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from chronos import Chronos2Pipeline

from forecasting_evaluation.config import Chronos2ModelConfig
from forecasting_evaluation.models.base import BasePredictionModel


class Chronos2Model(BasePredictionModel):
    """Wrapper for Amazon Chronos-2 forecasting model.

    Chronos-2 is a pretrained time series forecasting model that supports:
    - Multivariate forecasting
    - Past and future covariates
    - Probabilistic forecasting with quantiles
    """

    def __init__(self, config: Chronos2ModelConfig | None = None, seed: int = 42, **kwargs):
        """Initialize Chronos2Model.

        Parameters
        ----------
        device_map : str, optional
            Device mapping for model, by default "auto"
        torch_dtype : torch.dtype, optional
            Data type for model, by default torch.bfloat16
        **kwargs
            Additional arguments passed to BaseChronosPipeline.from_pretrained()
        """
        self.config = config or Chronos2ModelConfig()
        self.seed = seed
        self.quantile_levels = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dtype=float,
        )
        device_map = self._resolve_device_map()
        model_name_or_path = self._resolve_model_name_or_path()
        torch_dtype = self._resolve_torch_dtype(device_map)

        self.pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        # Chronos-2 truncates context to its configured length internally and
        # derives its instance-norm scaling from the truncated window, so feeding
        # a longer history is wasted work with no effect on the output. Read the
        # model's true context length and slice to it in predict() (the harness
        # now hands models the full prefix; each model self-windows, like Toto).
        self.context_length: int = int(self.pipeline.model_context_length)
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Loaded Chronos-2 pipeline from %s (device=%s, dtype=%s)",
            model_name_or_path,
            device_map,
            torch_dtype,
        )

    def _resolve_device_map(self) -> str:
        device = self.config.device
        if device == "auto":
            # Place the whole model on a single CUDA device explicitly. HF's
            # device_map="auto" runs accelerate's memory probe, which on these
            # nodes intermittently reports "Device 0 seems unavailable" and
            # silently falls back to CPU (≈100x slower). An explicit "cuda"
            # skips that probe; fall back to cpu only when CUDA is truly absent.
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logging.getLogger(__name__).warning(
                "model.chronos2.device is cuda but CUDA is unavailable; falling back to cpu"
            )
            return "cpu"
        return device

    def _resolve_torch_dtype(self, device_map: str) -> torch.dtype | str:
        dtype_name = self.config.torch_dtype
        if dtype_name == "auto":
            return torch.float32 if device_map == "cpu" else torch.bfloat16
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[dtype_name]

    def _resolve_model_name_or_path(self) -> str:
        if self.config.checkpoint_path:
            return str(self._resolve_local_checkpoint_path(self.config.checkpoint_path))

        if self.config.training_output_dir:
            return str(self._resolve_finetuned_checkpoint_dir(self.config.training_output_dir))

        return self.config.pretrained_model_name_or_path

    def _resolve_local_checkpoint_path(self, checkpoint_path: str) -> Path:
        local_path = Path(checkpoint_path).expanduser()
        if not local_path.exists():
            raise FileNotFoundError(f"Chronos-2 checkpoint not found: {local_path}")

        if local_path.is_dir() and not self._looks_like_chronos_checkpoint_dir(local_path):
            return self._resolve_finetuned_checkpoint_dir(str(local_path))

        return local_path

    def _resolve_finetuned_checkpoint_dir(self, training_output_dir: str) -> Path:
        output_dir = Path(training_output_dir).expanduser()
        if not output_dir.exists():
            raise FileNotFoundError(f"Chronos-2 training output directory not found: {output_dir}")

        checkpoint_name = self.config.finetuned_ckpt_name or self._load_finetuned_ckpt_name(
            output_dir
        )
        checkpoint_dir = output_dir / checkpoint_name
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Chronos-2 fine-tuned checkpoint directory not found: {checkpoint_dir}"
            )
        return checkpoint_dir

    def _load_finetuned_ckpt_name(self, output_dir: Path) -> str:
        for config_path in (
            output_dir / "training_config.json",
            output_dir / "training_config.yaml",
        ):
            if not config_path.exists():
                continue

            data = self._read_config_file(config_path)
            if not isinstance(data, dict):
                continue

            output_config = data.get("output", {})
            if isinstance(output_config, dict):
                checkpoint_name = output_config.get("finetuned_ckpt_name")
                if checkpoint_name:
                    return str(checkpoint_name)

        return "finetuned-ckpt"

    def _looks_like_chronos_checkpoint_dir(self, directory: Path) -> bool:
        return (directory / "config.json").exists() and any(
            (directory / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )

    def _read_config_file(self, config_path: Path) -> dict | None:
        if config_path.suffix == ".json":
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        try:
            import yaml
        except ImportError:
            logging.getLogger(__name__).warning(
                "PyYAML not available; cannot read %s",
                config_path,
            )
            return None

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        past_covariates: dict[str, np.ndarray] | None = None,
        future_covariates: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Generate forecasts for the given time series.

        Parameters
        ----------
        history : np.ndarray
            Full-prefix history of shape (n_features, history_length), may contain NaN.
        horizon : int
            Number of future hours to forecast.
        past_covariates : dict[str, np.ndarray] | None, optional
            Optional past-only covariates (forwarded by the harness if declared).
        future_covariates : dict[str, np.ndarray] | None, optional
            Optional covariates spanning history + horizon.

        Returns:
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            - point_result: Point predictions of shape (n_features, prediction_length)
            - quantiles_result: Quantile predictions of shape (n_features, prediction_length, n_quantiles)
              or None if quantiles cannot be computed
        """
        # Trim to the model's context window before handing the array to the
        # pipeline. Chronos-2 truncates to ``context_length`` and derives its
        # instance-norm scaling from that window internally, so this is
        # output-identical to passing the full prefix — but avoids preprocessing
        # multi-year arrays on every window (the post-7939848 harness passes the
        # full prefix; models self-window, as Toto does). Gated on no covariates
        # so target/covariate alignment stays trivially correct (the current eval
        # passes none); with covariates present we hand over the full prefix and
        # let the pipeline truncate target+covariates consistently.
        target = history
        if (
            past_covariates is None
            and future_covariates is None
            and history.shape[1] > self.context_length
        ):
            target = history[:, -self.context_length :]
        prediction_length = horizon

        # Construct input in the format expected by Chronos2Pipeline
        model_inputs = {
            "target": target,
        }

        # Add covariates if provided
        if past_covariates is not None:
            model_inputs["past_covariates"] = past_covariates
        if future_covariates is not None:
            model_inputs["future_covariates"] = future_covariates

        # Call pipeline.predict_quantiles.
        # quantiles: list[(n_variates, prediction_length, n_quantiles)]
        # mean: list[(n_variates, prediction_length)]
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=[model_inputs],
            prediction_length=prediction_length,
            quantile_levels=self.quantile_levels.tolist(),
        )

        quantiles_result = quantiles[0].cpu().numpy()
        point_result = mean[0].cpu().numpy()

        return point_result, quantiles_result
