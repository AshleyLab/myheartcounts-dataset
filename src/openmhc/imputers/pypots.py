"""PyPOTS-backed imputers (BRITS, TimesNet, DLinear, FEDformer).

Each public class loads a pre-trained PyPOTS checkpoint and exposes the
``Imputer`` protocol used by :func:`openmhc.evaluate_imputation`. Training
is out of scope — supply a ``.pypots`` file produced by PyPOTS's
``model.save()``.

Checkpoints
-----------
The architecture hyperparameters passed to the imputer's constructor must
match the values used when the model was trained. PyPOTS's ``load()`` is
an instance method: the wrapper builds the model with the matching arch
args, then loads the weights file. Mismatched args surface as a torch
``RuntimeError`` (size mismatch) at construction time.

``model_path`` accepts either a direct ``.pypots`` file or a directory
containing one (the first match is used).

Normalization
-------------
Pass ``normalization_stats_path`` pointing at a ``normalization_stats.json``
(produced by the H5 export) when the model was trained on z-scored data.
The wrapper applies the same z-score before inference and inverts it on
the way out. Without this argument, raw data is passed to the model.

Install
-------
``pip install openmhc[pypots]`` to pull in the PyPOTS dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from openmhc.imputers._base import BaseImputer
from openmhc.imputers._release import ReleaseLoadableMixin


class _PyPOTSImputerBase(ReleaseLoadableMixin, BaseImputer):
    """Shared machinery for PyPOTS-backed imputers.

    Subclasses set the class-level ``model_name`` attribute and implement
    :meth:`_build_model`, which returns an instantiated (but not yet
    weight-loaded) PyPOTS model. The base class handles file resolution,
    normalization, the channels-first ↔ time-first transpose, and writing
    the model's output only into ``target_mask == 1`` positions.

    ``from_release`` is inherited from :class:`ReleaseLoadableMixin`.
    """

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        device: str = "cuda",
        inference_batch_size: int = 64,
        normalization_stats_path: str | Path | None = None,
        n_steps: int = 1440,
        n_features: int = 19,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(version=version, data_dir=data_dir)
        self._device = device
        self._inference_batch_size = int(inference_batch_size)
        self._n_steps = int(n_steps)
        self._n_features = int(n_features)
        self._model_file = self._resolve_model_file(Path(model_path))
        self._stats = self._load_stats(normalization_stats_path)
        self._model = self._build_model()
        self._model.load(str(self._model_file))
        self.name = f"pypots_{self.model_name}"

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _build_model(self):
        raise NotImplementedError(
            "Subclasses must implement `_build_model` to return a PyPOTS model "
            "instantiated with matching architecture hyperparameters."
        )

    # ------------------------------------------------------------------
    # Checkpoint + stats helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_file(model_path: Path) -> Path:
        if not model_path.exists():
            raise FileNotFoundError(f"PyPOTS model path does not exist: {model_path}")
        if model_path.is_file():
            return model_path
        matches = sorted(model_path.glob("*.pypots"))
        if not matches:
            raise FileNotFoundError(
                f"No .pypots checkpoint found under directory {model_path}"
            )
        return matches[0]

    @staticmethod
    def _load_stats(path: str | Path | None) -> dict | None:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {p}")
        raw = json.loads(p.read_text())
        return {
            "means": np.asarray(raw["means"], dtype=np.float32),
            "stds": np.asarray(raw["stds"], dtype=np.float32),
            "channels": tuple(int(c) for c in raw["channels"]),
            "epsilon": float(raw.get("epsilon", 1e-8)),
        }

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        s = self._stats
        if s is None:
            return out
        eps = s["epsilon"]
        for ch in s["channels"]:
            out[..., ch, :] = (out[..., ch, :] - s["means"][ch]) / (s["stds"][ch] + eps)
        return out

    def _denormalize(self, z: np.ndarray) -> np.ndarray:
        out = z.copy()
        s = self._stats
        if s is None:
            return out
        eps = s["epsilon"]
        for ch in s["channels"]:
            out[..., ch, :] = out[..., ch, :] * (s["stds"][ch] + eps) + s["means"][ch]
        return out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        work = self._normalize(data) if self._stats is not None else data.copy()
        target_bool = target_mask > 0.5
        work[target_bool] = np.nan

        time_first = np.transpose(work, (0, 2, 1))
        out = self._model.impute({"X": time_first})
        if isinstance(out, dict):
            out = out["imputation"]
        imputed = np.transpose(np.asarray(out), (0, 2, 1))
        if self._stats is not None:
            imputed = self._denormalize(imputed)

        result = data.copy()
        result[target_bool] = imputed[target_bool]
        return result.astype(np.float32, copy=False)


class BRITSImputer(_PyPOTSImputerBase):
    """PyPOTS BRITS (Bidirectional Recurrent Imputation for Time Series).

    Args:
        model_path: Path to a ``.pypots`` checkpoint or a directory holding one.
        rnn_hidden_size: Must match training. Sole BRITS-specific arch arg.
        device: Torch device (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        inference_batch_size: Batch size for PyPOTS internal inference loop.
        normalization_stats_path: Optional path to a stats JSON. If supplied,
            inputs are z-scored before inference and predictions are
            denormalized on the way out.
        n_steps: Sequence length the model was trained on (1440 for 1-day
            windows, 10080 for 7-day windows).
        n_features: Number of channels (19 in this benchmark).
        data_dir: Override for the openmhc dataset root.
    """

    model_name = "brits"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        rnn_hidden_size: int = 128,
        device: str = "cuda",
        inference_batch_size: int = 64,
        normalization_stats_path: str | Path | None = None,
        n_steps: int = 1440,
        n_features: int = 19,
        data_dir: str | Path | None = None,
    ) -> None:
        self._rnn_hidden_size = int(rnn_hidden_size)
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            normalization_stats_path=normalization_stats_path,
            n_steps=n_steps,
            n_features=n_features,
            data_dir=data_dir,
        )

    def _build_model(self):
        from pypots.imputation import BRITS  # lazy

        return BRITS(
            n_steps=self._n_steps,
            n_features=self._n_features,
            rnn_hidden_size=self._rnn_hidden_size,
            batch_size=self._inference_batch_size,
            device=self._device,
        )


class TimesNetImputer(_PyPOTSImputerBase):
    """PyPOTS TimesNet imputer.

    Args:
        model_path: Path to a ``.pypots`` checkpoint or a directory holding one.
        n_layers: Number of TimesBlock layers (must match training).
        top_k: Top-k frequencies used for period decomposition.
        d_model: Model dimension.
        d_ffn: Feed-forward inner dimension.
        n_kernels: Number of inception kernels per TimesBlock.
        dropout: Dropout rate.
        apply_nonstationary_norm: Toggle for non-stationary normalization.
        device, inference_batch_size, normalization_stats_path, n_steps,
        n_features, data_dir: See :class:`BRITSImputer`.
    """

    model_name = "timesnet"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        n_layers: int = 2,
        top_k: int = 5,
        d_model: int = 64,
        d_ffn: int = 64,
        n_kernels: int = 6,
        dropout: float = 0.1,
        apply_nonstationary_norm: bool = False,
        device: str = "cuda",
        inference_batch_size: int = 64,
        normalization_stats_path: str | Path | None = None,
        n_steps: int = 1440,
        n_features: int = 19,
        data_dir: str | Path | None = None,
    ) -> None:
        self._n_layers = int(n_layers)
        self._top_k = int(top_k)
        self._d_model = int(d_model)
        self._d_ffn = int(d_ffn)
        self._n_kernels = int(n_kernels)
        self._dropout = float(dropout)
        self._apply_nonstationary_norm = bool(apply_nonstationary_norm)
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            normalization_stats_path=normalization_stats_path,
            n_steps=n_steps,
            n_features=n_features,
            data_dir=data_dir,
        )

    def _build_model(self):
        from pypots.imputation import TimesNet  # lazy

        return TimesNet(
            n_steps=self._n_steps,
            n_features=self._n_features,
            n_layers=self._n_layers,
            top_k=self._top_k,
            d_model=self._d_model,
            d_ffn=self._d_ffn,
            n_kernels=self._n_kernels,
            dropout=self._dropout,
            apply_nonstationary_norm=self._apply_nonstationary_norm,
            batch_size=self._inference_batch_size,
            device=self._device,
        )


class DLinearImputer(_PyPOTSImputerBase):
    """PyPOTS DLinear imputer (decomposition-based linear model).

    Args:
        model_path: Path to a ``.pypots`` checkpoint or a directory holding one.
        moving_avg_window_size: Window size for the trend/seasonal decomposition.
        d_model: Embedding dimension. Required when ``individual=False`` (the
            default); ignored when ``individual=True``.
        individual: ``True`` puts a separate linear head per feature.
        device, inference_batch_size, normalization_stats_path, n_steps,
        n_features, data_dir: See :class:`BRITSImputer`.
    """

    model_name = "dlinear"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        moving_avg_window_size: int = 25,
        d_model: int | None = 64,
        individual: bool = False,
        device: str = "cuda",
        inference_batch_size: int = 64,
        normalization_stats_path: str | Path | None = None,
        n_steps: int = 1440,
        n_features: int = 19,
        data_dir: str | Path | None = None,
    ) -> None:
        self._moving_avg_window_size = int(moving_avg_window_size)
        self._d_model = d_model if d_model is None else int(d_model)
        self._individual = bool(individual)
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            normalization_stats_path=normalization_stats_path,
            n_steps=n_steps,
            n_features=n_features,
            data_dir=data_dir,
        )

    def _build_model(self):
        from pypots.imputation import DLinear  # lazy

        return DLinear(
            n_steps=self._n_steps,
            n_features=self._n_features,
            moving_avg_window_size=self._moving_avg_window_size,
            individual=self._individual,
            d_model=self._d_model,
            batch_size=self._inference_batch_size,
            device=self._device,
        )


class FEDformerImputer(_PyPOTSImputerBase):
    """PyPOTS FEDformer imputer (frequency-enhanced decomposed Transformer).

    Args:
        model_path: Path to a ``.pypots`` checkpoint or a directory holding one.
        version: OpenMHC dataset version (``"xs"`` or ``"full"``). Distinct
            from the FEDformer architectural variant — see ``variant``.
        n_layers: Transformer layers.
        d_model: Model dimension.
        n_heads: Attention heads.
        d_ffn: Feed-forward inner dimension.
        moving_avg_window_size: Trend decomposition window.
        dropout: Dropout rate.
        variant: FEDformer frequency basis. ``"Fourier"`` or ``"Wavelets"``.
            Maps to PyPOTS's own ``version`` kwarg internally.
        modes: Number of frequency modes.
        mode_select: ``"random"`` or ``"low"``.
        device, inference_batch_size, normalization_stats_path, n_steps,
        n_features, data_dir: See :class:`BRITSImputer`.
    """

    model_name = "fedformer"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        n_layers: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        d_ffn: int = 64,
        moving_avg_window_size: int = 25,
        dropout: float = 0.1,
        variant: Literal["Fourier", "Wavelets"] = "Fourier",
        modes: int = 32,
        mode_select: Literal["random", "low"] = "random",
        device: str = "cuda",
        inference_batch_size: int = 64,
        normalization_stats_path: str | Path | None = None,
        n_steps: int = 1440,
        n_features: int = 19,
        data_dir: str | Path | None = None,
    ) -> None:
        self._n_layers = int(n_layers)
        self._d_model = int(d_model)
        self._n_heads = int(n_heads)
        self._d_ffn = int(d_ffn)
        self._moving_avg_window_size = int(moving_avg_window_size)
        self._dropout = float(dropout)
        self._variant = variant
        self._modes = int(modes)
        self._mode_select = mode_select
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            normalization_stats_path=normalization_stats_path,
            n_steps=n_steps,
            n_features=n_features,
            data_dir=data_dir,
        )

    def _build_model(self):
        from pypots.imputation import FEDformer  # lazy

        return FEDformer(
            n_steps=self._n_steps,
            n_features=self._n_features,
            n_layers=self._n_layers,
            d_model=self._d_model,
            n_heads=self._n_heads,
            d_ffn=self._d_ffn,
            moving_avg_window_size=self._moving_avg_window_size,
            dropout=self._dropout,
            version=self._variant,
            modes=self._modes,
            mode_select=self._mode_select,
            batch_size=self._inference_batch_size,
            device=self._device,
        )
