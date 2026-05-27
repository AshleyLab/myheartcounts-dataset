"""Generic adapter that wraps a pre-trained ``torch.nn.Module`` as an Imputer.

Handles the boilerplate that's common to almost every neural imputer:
numpy↔torch conversion, NaN replacement, mini-batching across N for
GPU memory, per-channel z-score normalization on the continuous
channels, sigmoid on the binary channels, and copy-back into only the
``target_mask == 1`` positions.

Users supply a ``torch.nn.Module`` and (optionally) the shape conventions
their model expects. Training happens entirely outside this class — you
load weights in your own script, then pass the model in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from openmhc.imputers._base import BaseImputer


class TorchImputer(BaseImputer):
    """Wrap a pre-trained ``nn.Module`` as an Imputer.

    The wrapper assumes the model takes a batched tensor of imputable
    inputs (with NaNs replaced) and optionally a "valid" mask, and
    returns reconstructions for every position. Only positions where
    ``target_mask == 1`` are written into the output; everything else
    is left as-is.

    Args:
        model: A ``torch.nn.Module`` with weights already loaded. The
            wrapper sets it to ``eval()`` and moves it to ``device``.
        device: Torch device (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        inference_batch_size: Inner mini-batch size; the wrapper splits
            the outer batch into chunks of this size to bound GPU memory.
        channels_first: ``True`` if the model wants shape
            ``(B, C, T)``; ``False`` if ``(B, T, C)``.
        nan_fill: How to fill NaNs in the model input. ``"zero"`` (the
            simplest) or ``"channel_mean"`` (uses training-set means).
        normalize: If ``True``, z-score the continuous channels
            (``0..len(binary_channels[0])-1``) using training stats
            before the forward pass and denormalize predictions
            afterwards. Binary channels (default 7-18) pass through.
        binary_channels: Channel indices treated as binary. Sigmoid is
            applied to the model output for these channels so the result
            is in ``[0, 1]``.
        forward_signature: How to call the model.

            - ``"x"``: ``model(x)`` — model gets only the (filled, possibly
              normalized) input tensor.
            - ``"x_mask"``: ``model(x, valid_mask)`` — model also gets
              a binary mask tensor (same shape as ``x``) where ``1`` means
              the model may look at that position.
        model_name: Optional human-readable name for result labeling.
            Defaults to the model class name.
        data_dir: Override for the dataset root.
    """

    def __init__(
        self,
        model,
        version,
        device: str = "cuda",
        inference_batch_size: int = 128,
        channels_first: bool = True,
        nan_fill: Literal["zero", "channel_mean"] = "channel_mean",
        normalize: bool = True,
        binary_channels: tuple[int, ...] = tuple(range(7, 19)),
        forward_signature: Literal["x", "x_mask"] = "x_mask",
        model_name: str | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        import torch  # local import — torch is a heavy dep

        super().__init__(version=version, data_dir=data_dir)
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._inference_batch_size = inference_batch_size
        self._channels_first = channels_first
        self._nan_fill = nan_fill
        self._normalize = normalize
        self._binary_channels = tuple(binary_channels)
        self._forward_signature = forward_signature

        if normalize or nan_fill == "channel_mean":
            self._means, self._stds = self.compute_channel_means_stds()
        else:
            self._means = np.zeros(self.n_channels, dtype=np.float32)
            self._stds = np.ones(self.n_channels, dtype=np.float32)

        self.name = model_name or type(model).__name__

    @property
    def _continuous_channels(self) -> np.ndarray:
        """Channel indices not in ``binary_channels``."""
        all_idx = np.arange(self.n_channels)
        binary = np.array(self._binary_channels, dtype=int)
        return np.setdiff1d(all_idx, binary, assume_unique=True)

    def _prepare_input(
        self, data: np.ndarray, observed_mask: np.ndarray, target_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replace NaNs (fill), optionally normalize, return (x, valid_mask)."""
        valid_mask = (observed_mask > 0.5) & (target_mask < 0.5)
        valid_mask = valid_mask.astype(np.float32)

        if self._nan_fill == "channel_mean":
            fill_values = self._means[None, :, None]
        else:
            fill_values = np.zeros((1, self.n_channels, 1), dtype=np.float32)
        x = np.where(valid_mask > 0.5, data, fill_values).astype(np.float32)

        if self._normalize:
            cont = self._continuous_channels
            if cont.size > 0:
                means = self._means[None, :, None]
                stds = self._stds[None, :, None]
                x[:, cont, :] = (x[:, cont, :] - means[:, cont, :]) / stds[:, cont, :]
        return x, valid_mask

    def _denormalize(self, predictions: np.ndarray) -> np.ndarray:
        """Reverse normalization on continuous channels; sigmoid the binary ones."""
        out = predictions.astype(np.float32, copy=True)
        if self._normalize:
            cont = self._continuous_channels
            if cont.size > 0:
                means = self._means[None, :, None]
                stds = self._stds[None, :, None]
                out[:, cont, :] = out[:, cont, :] * stds[:, cont, :] + means[:, cont, :]
        if len(self._binary_channels) > 0:
            bin_idx = np.array(self._binary_channels, dtype=int)
            out[:, bin_idx, :] = 1.0 / (1.0 + np.exp(-out[:, bin_idx, :]))
        return out

    def _forward_chunk(self, x_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        torch = self._torch
        x_t = torch.from_numpy(x_np).to(self._device)
        mask_t = torch.from_numpy(mask_np).to(self._device)
        if not self._channels_first:
            x_t = x_t.transpose(1, 2)
            mask_t = mask_t.transpose(1, 2)

        if self._forward_signature == "x_mask":
            y_t = self._model(x_t, mask_t)
        else:
            y_t = self._model(x_t)

        if not self._channels_first:
            y_t = y_t.transpose(1, 2)
        return y_t.detach().cpu().numpy()

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        x, valid_mask = self._prepare_input(data, observed_mask, target_mask)
        N = x.shape[0]
        bs = max(1, int(self._inference_batch_size))
        preds_chunks = []
        with torch.no_grad():
            for start in range(0, N, bs):
                end = min(start + bs, N)
                preds_chunks.append(self._forward_chunk(x[start:end], valid_mask[start:end]))
        preds = np.concatenate(preds_chunks, axis=0)
        preds = self._denormalize(preds)

        result = data.copy()
        fill_positions = target_mask > 0.5
        result[fill_positions] = preds[fill_positions]
        return result.astype(np.float32, copy=False)
