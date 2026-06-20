"""Toto 1.0 foundational forecasting model wrapper.

Integrates Datadog/Toto-Open-Base-1.0 for zero-shot multivariate
time series forecasting with probabilistic (Student-T mixture) outputs.

Requires: pip install toto-ts
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from forecasting_evaluation.models.base import BasePredictionModel

logger = logging.getLogger(__name__)


@dataclass
class TotoModelConfig:
    """Toto model hyperparameters."""

    pretrained_model_name_or_path: str = "Datadog/Toto-Open-Base-1.0"
    checkpoint_path: str | None = None
    lora_alpha: float | None = None
    device: str = "cuda"
    context_length: int = 2048
    num_samples: int = 256
    samples_per_batch: int = 256
    use_kv_cache: bool = True
    time_interval_seconds: int = 3600  # 1 hour for hourly wearable data


class TotoModel(BasePredictionModel):
    """Wrapper for Datadog Toto 1.0 zero-shot forecasting model.

    Toto is a 151M-parameter transformer pretrained on massive time series
    corpora. It outputs a Student-T mixture distribution, from which we
    extract median point forecasts and quantile predictions.
    """

    def __init__(self, config: TotoModelConfig | None = None, seed: int = 42):
        """Initialize the Toto wrapper and load model weights."""
        self.config = config or TotoModelConfig()
        self.seed = seed
        self.quantile_levels = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dtype=float,
        )

        try:
            from toto.data.util.dataset import (
                MaskedTimeseries,
                pad_array,
                pad_id_mask,
                replace_extreme_values,
            )
            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto
        except ImportError as e:
            raise ImportError(
                "toto-ts is required for the Toto model. Install with: pip install toto-ts"
            ) from e

        self._MaskedTimeseries = MaskedTimeseries
        self._pad_array = pad_array
        self._pad_id_mask = pad_id_mask
        self._replace_extreme_values = replace_extreme_values
        self.device = self._resolve_device()

        logger.info(
            "Loading Toto from %s (checkpoint=%s, device=%s)",
            self.config.pretrained_model_name_or_path,
            self.config.checkpoint_path or "none",
            self.device,
        )
        toto_wrapper = Toto.from_pretrained(self.config.pretrained_model_name_or_path)
        if self.config.checkpoint_path:
            self._load_lightning_checkpoint(toto_wrapper, self.config.checkpoint_path)
        backbone = toto_wrapper.model.to(self.device)
        self.forecaster = TotoForecaster(model=backbone)
        logger.info(
            "Loaded Toto model (context_length=%d, num_samples=%d)",
            self.config.context_length,
            self.config.num_samples,
        )

    def _resolve_device(self) -> str:
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("model.toto.device is cuda but CUDA is unavailable; falling back to cpu")
            return "cpu"
        return device

    def _autocast(self):
        """bfloat16 autocast on CUDA (≈2x faster); no-op on CPU.

        Toto's reference eval runs the backbone in reduced precision; the
        Student-T forecast distribution is unaffected beyond Monte-Carlo noise.
        """
        if str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _load_lightning_checkpoint(self, toto_wrapper, checkpoint_path: str) -> None:
        """Load a local Lightning Toto checkpoint into the HuggingFace Toto wrapper.

        The fine-tuning checkpoints produced by Lightning store weights under a
        ``model.base_model.`` prefix. LoRA-trained checkpoints additionally store
        ``base_layer`` plus ``lora_A/lora_B`` weights; for inference we merge the
        LoRA delta into the corresponding linear layer before loading.
        """
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Toto checkpoint not found: {checkpoint}")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported Toto checkpoint format: {checkpoint}")

        merged_state = self._convert_lightning_state_dict(state_dict)
        incompatible = toto_wrapper.load_state_dict(merged_state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.endswith("rotary_emb.freqs")]
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            logger.warning(
                "Toto checkpoint load left %d missing keys; first keys: %s",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "Toto checkpoint load ignored %d unexpected keys; first keys: %s",
                len(unexpected),
                unexpected[:5],
            )
        logger.info("Loaded Toto Lightning checkpoint from %s", checkpoint)

    def _convert_lightning_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert Lightning/LoRA key layout to Toto wrapper state_dict layout."""
        converted: dict[str, torch.Tensor] = {}
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}

        for raw_key, value in state_dict.items():
            key = self._strip_lightning_toto_prefix(raw_key)
            if ".lora_A.default.weight" in key:
                base_key = key.replace(".lora_A.default.weight", ".weight")
                lora_a[base_key] = value.detach().cpu()
                continue
            if ".lora_B.default.weight" in key:
                base_key = key.replace(".lora_B.default.weight", ".weight")
                lora_b[base_key] = value.detach().cpu()
                continue
            key = key.replace(".base_layer.weight", ".weight")
            key = key.replace(".base_layer.bias", ".bias")
            converted[key] = value.detach().cpu()

        for base_key, a_weight in lora_a.items():
            b_weight = lora_b.get(base_key)
            base_weight = converted.get(base_key)
            if b_weight is None or base_weight is None:
                logger.warning("Skipping incomplete Toto LoRA weights for %s", base_key)
                continue
            rank = int(a_weight.shape[0])
            alpha = (
                float(self.config.lora_alpha) if self.config.lora_alpha is not None else float(rank)
            )
            scaling = alpha / max(rank, 1)
            converted[base_key] = (
                base_weight + (b_weight @ a_weight).to(base_weight.dtype) * scaling
            )

        return converted

    @staticmethod
    def _strip_lightning_toto_prefix(key: str) -> str:
        """Normalize common Lightning wrapper prefixes for Toto checkpoints."""
        for prefix in ("model.base_model.", "base_model."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Generate zero-shot forecasts for the given history window.

        Args:
            history: Full-prefix history of shape (n_features, history_length),
                may contain NaN.
            horizon: Number of future hours to forecast.

        Returns:
            Tuple of:
                - point_result: Median forecast, shape (n_features, prediction_length).
                - quantiles_result: Quantile forecasts, shape
                    (n_features, prediction_length, n_quantiles).
        """
        prediction_length = horizon
        n_features, history_length = history.shape

        # Trim history to configured context window
        ctx = min(history_length, self.config.context_length)
        history = history[:, -ctx:]

        series = torch.tensor(history, dtype=torch.float32, device=self.device)

        # NaN values must be replaced with 0 and masked out via padding_mask.
        # If NaN propagates into the model's Student-T distribution parameters,
        # it causes GreaterThan constraint errors.
        padding_mask = ~torch.isnan(series)
        series = torch.nan_to_num(series, nan=0.0)

        id_mask = torch.zeros_like(series, dtype=torch.float32)
        timestamp_seconds = torch.zeros(
            n_features,
            ctx,
            dtype=torch.float32,
            device=self.device,
        )
        time_interval_seconds = torch.full(
            (n_features,),
            fill_value=float(self.config.time_interval_seconds),
            dtype=torch.float32,
            device=self.device,
        )

        masked_ts = self._MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # When the horizon fits inside a single patch, Toto's autoregressive
        # decoder runs exactly one step, so all `num_samples` draws are i.i.d.
        # samples from one predicted distribution. We can encode the context
        # once and draw every sample from that distribution instead of
        # replicating the context `num_samples` times — ~50x faster, same
        # distribution (verified: difference is pure Monte-Carlo noise).
        patch_size = self.forecaster.model.patch_embed.patch_size
        if prediction_length <= patch_size:
            samples = self._encode_once_samples(masked_ts, prediction_length, patch_size)
        else:
            with torch.no_grad(), self._autocast():
                forecast = self.forecaster.forecast(
                    masked_ts,
                    prediction_length=prediction_length,
                    num_samples=self.config.num_samples,
                    samples_per_batch=self.config.samples_per_batch,
                    use_kv_cache=self.config.use_kv_cache,
                )
            samples = forecast.samples  # (batch, variate, horizon, samples)

        # samples: (batch, variate, horizon, samples). We feed a single
        # sub-trajectory, so squeeze the batch dim to match the
        # (n_features, prediction_length) contract expected downstream.
        # point forecast: median across samples.
        point_result = samples.quantile(0.5, dim=-1).float().cpu().numpy()
        if point_result.ndim == 3 and point_result.shape[0] == 1:
            point_result = point_result[0]

        # quantile forecasts -> (n_features, prediction_length, n_quantiles)
        quantiles_result = np.stack(
            [
                samples.quantile(float(q), dim=-1).float().cpu().numpy()
                for q in self.quantile_levels
            ],
            axis=-1,
        )
        if quantiles_result.ndim == 4 and quantiles_result.shape[0] == 1:
            quantiles_result = quantiles_result[0]

        return point_result, quantiles_result

    def _encode_once_samples(
        self,
        masked_ts,
        prediction_length: int,
        patch_size: int,
    ) -> torch.Tensor:
        """Single-forward sampling for horizons that fit one patch.

        Encodes the context once and draws ``num_samples`` from the resulting
        one-step distribution. Mirrors the per-step logic of
        ``TotoForecaster.generate_samples`` (last-patch slice plus
        ``replace_extreme_values``) without the per-sample context replication.

        Returns:
            Sample tensor shaped (batch, variate, horizon, num_samples).
        """
        forecaster = self.forecaster
        model = forecaster.model
        stride = model.patch_embed.stride

        batch = torch.utils.data.default_collate([masked_ts])
        series = self._pad_array(batch.series, stride)
        padding_mask = self._pad_array(batch.padding_mask, stride)
        id_mask = batch.id_mask
        if id_mask is not None:
            id_mask = self._pad_id_mask(id_mask, stride)

        with torch.no_grad(), self._autocast():
            base_distr, loc, scale = model(
                inputs=series,
                input_padding_mask=padding_mask,
                id_mask=id_mask,
                kv_cache=None,
                scaling_prefix_length=series.shape[-1],
                num_exogenous_variables=0,
            )
            distr = forecaster.create_affine_transformed(base_distr, loc, scale)
            # (num_samples, batch, variate, series_len) -> last predicted patch
            draws = distr.sample((self.config.num_samples,))
            draws = self._replace_extreme_values(draws[..., -patch_size:])

        # trim to horizon and move samples to the trailing axis:
        # (S, B, V, patch) -> (B, V, horizon, S)
        draws = draws[..., :prediction_length].permute(1, 2, 3, 0)
        return draws
