"""LSM2 (Latent Sequence Model v2) imputer wrappers.

Loads a pre-trained Lightning ``.ckpt`` produced by training the
:class:`openmhc.models.lsm2.modules.LSM2Module` or
:class:`openmhc.models.lsm2.modules.WeeklySparseLSM2Module`, and adapts it to
the :class:`Imputer` protocol used by :func:`openmhc.evaluate_imputation`.

Two public classes are exposed:

- :class:`LSM2Imputer` — daily (``seq_length=1440, patch_size=10``) and weekly
  (``seq_length=10080, patch_size=60``) variants share one model class and one
  wrapper; pass the matching sizing as keyword args.
- :class:`LSM2WeeklySparseImputer` — wraps the distinct per-day-encoder +
  sparse-cross-day-decoder model used for the weekly-sparse paper variant.

Install
-------
``pip install openmhc[lsm2]`` to pull in ``pytorch-lightning``.

Inference flow (shared)
-----------------------
1. z-score channels 0-6 with a sibling ``normalization_stats.json`` (same
   JSON format used by the PyPOTS wrappers; identity stats for binary
   channels 7-18 pass through).
2. Replace remaining NaNs (naturally missing + artificially-masked target
   positions) with 0 — this is the channel mean post-normalization.
3. Build a patch-level ``inherited_mask`` (1 = missing for the model)
   from the effective-valid mask ``observed_mask * (1 - target_mask)``,
   with the special HR-channel-5 rule: only mark a patch missing if
   *all* minutes in the patch are missing.
4. Run a custom inference forward pass that bypasses training-time
   artificial masking (``total_mask = inherited_mask``).
5. Unpatchify, optional sigmoid on binary channels (if
   ``model.use_hybrid_loss``), denormalize.
6. Write predictions back only at ``target_mask == 1`` positions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from openmhc._device import resolve_device
from openmhc.imputers._base import BaseImputer
from openmhc.imputers._release import ReleaseLoadableMixin

# Heart-rate channel index — the inherited-mask construction treats this
# channel asymmetrically (a patch is only marked missing if *all* minutes
# are missing, vs the "any minute missing" rule for every other channel).
_HR_CHANNEL = 5


class _LSM2ImputerBase(ReleaseLoadableMixin, BaseImputer):
    """Shared machinery for LSM2-backed imputers.

    Subclasses set the class-level ``model_name`` attribute, implement
    :meth:`_load_model` to return the bare ``nn.Module`` (with weights
    populated), and implement :meth:`_inference_forward` to run a custom
    forward pass that consumes the patch-level inherited mask.

    ``from_release`` is inherited from :class:`ReleaseLoadableMixin`.
    """

    model_name: str = ""

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        device: str = "auto",
        inference_batch_size: int = 64,
        inference_dropout_removal_ratio: float | None = 0.0,
        normalization_stats_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        import torch  # heavy dep — local import

        super().__init__(version=version, data_dir=data_dir)
        self._torch = torch
        self._device = torch.device(resolve_device(device))
        self._inference_batch_size = int(inference_batch_size)
        self._inference_dropout_removal_ratio = inference_dropout_removal_ratio
        self._ckpt_file = self._resolve_ckpt_file(Path(model_path))
        self._stats = self._load_stats(normalization_stats_path)
        self._model = self._load_model()
        self._model.to(self._device).eval()
        self.name = f"lsm2_{self.model_name}" if self.model_name else "lsm2"

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _load_model(self):
        raise NotImplementedError

    def _inference_forward(self, x, inherited_mask):
        """Custom forward pass with ``total_mask = inherited_mask``.

        Returns patch-space predictions of shape ``(B, num_patches, patch_size)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # File / stats resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ckpt_file(model_path: Path) -> Path:
        """Resolve to a single ``.ckpt`` file (direct, or first inside a dir)."""
        if not model_path.exists():
            raise FileNotFoundError(f"LSM2 checkpoint path does not exist: {model_path}")
        if model_path.is_file():
            return model_path
        # Prefer .ckpt; fall back to .pypots for compatibility with the
        # shared release layout (the manifest stores whatever filename was
        # given at build time).
        for pattern in ("*.ckpt", "*.pt", "*.pth"):
            matches = sorted(model_path.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"No .ckpt / .pt / .pth checkpoint found under directory {model_path}"
        )

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
    # Patch-level inherited mask
    # ------------------------------------------------------------------

    def _create_inherited_mask(self, valid_mask):
        """``valid_mask`` is a torch tensor (B, C, L); returns (B, num_patches).

        Patches are channel-major: ``[ch0_p0, ch0_p1, ..., ch1_p0, ...]``.
        A patch is "missing" (value 1) if ANY minute is unobserved — except
        for the heart-rate channel where the rule is ALL minutes missing.
        """
        torch = self._torch
        B, C, L = valid_mask.shape
        patch_size = int(self._model.patch_size)
        num_patches_per_channel = L // patch_size

        m = valid_mask.view(B, C, num_patches_per_channel, patch_size)
        is_missing = ~m.all(dim=-1)
        if 0 <= _HR_CHANNEL < C:
            hr_all_missing = ~m[:, _HR_CHANNEL, :, :].any(dim=-1)
            is_missing[:, _HR_CHANNEL, :] = hr_all_missing
        return is_missing.view(B, -1).float()

    # ------------------------------------------------------------------
    # Inference orchestration
    # ------------------------------------------------------------------

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        result = data.copy()
        N = data.shape[0]
        bs = max(1, self._inference_batch_size)

        # Effective valid mask: positions the model gets to see.
        valid = observed_mask * (1.0 - target_mask)

        for start in range(0, N, bs):
            end = min(start + bs, N)
            batch_data = data[start:end]
            batch_valid = valid[start:end]
            batch_target = target_mask[start:end]

            # 1. normalize (channels 0-6 z-score; binary identity)
            x_norm = self._normalize(batch_data)
            # 2. fill remaining NaNs with 0 (post-normalization mean)
            x_filled = np.where(np.isfinite(x_norm), x_norm, 0.0).astype(np.float32)

            x_t = torch.from_numpy(x_filled).to(self._device)
            valid_t = torch.from_numpy(batch_valid.astype(np.float32)).to(self._device)
            inherited_mask = self._create_inherited_mask(valid_t)

            with torch.no_grad():
                pred = self._inference_forward(x_t, inherited_mask)
                reconstructed = self._model.unpatchify(pred)
                if getattr(self._model, "use_hybrid_loss", False):
                    reconstructed[:, 7:, :] = torch.sigmoid(reconstructed[:, 7:, :])

            recon_np = reconstructed.detach().cpu().numpy()
            recon_np = self._denormalize(recon_np)

            tb = batch_target > 0.5
            result[start:end][tb] = recon_np[tb]

        return result.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Daily / weekly (standard MAE-ViT)
# ---------------------------------------------------------------------------


class LSM2Imputer(_LSM2ImputerBase):
    """LSM2 imputer for daily and weekly variants.

    Daily uses ``seq_length=1440, patch_size=10``; weekly uses
    ``seq_length=10080, patch_size=60``. Both share the same model class
    (:class:`openmhc.models.lsm2.LSM2ViT1D`), so the same wrapper handles
    both — pass the matching sizing as keyword arguments.

    Args:
        model_path: Path to a Lightning ``.ckpt`` file or a directory
            containing one.
        seq_length, patch_size, in_channels, embed_dim, depth, num_heads,
        decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio,
        mask_ratio: Architecture hyperparameters that match the trained
            model. These are recorded by the release manifest and accepted
            here for `from_release` splatting and documentation, but the
            actual weights and architecture are restored from the
            checkpoint's saved hparams — mismatched values are tolerated
            by Lightning's ``load_from_checkpoint``.
        device: Torch device.
        inference_batch_size: Inner mini-batch size.
        inference_dropout_removal_ratio: Override the checkpoint's
            ``dropout_removal_ratio`` for the prioritized-keep step.
            Defaults to ``0.0`` (no token removal — deterministic
            inference). Pass ``None`` to fall back to the checkpoint's
            ``dropout_removal_ratio`` (nondeterministic; uses an
            unseeded ``torch.rand``).
        normalization_stats_path: Optional sibling JSON with z-score
            stats. Should match the training-time stats.
        data_dir: Override for the openmhc dataset root.
    """

    model_name = "lsm2"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        device: str = "auto",
        inference_batch_size: int = 64,
        inference_dropout_removal_ratio: float | None = 0.0,
        normalization_stats_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        **_extra: Any,
    ) -> None:
        self._declared_arch = {
            "seq_length": seq_length,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
            "mlp_ratio": mlp_ratio,
            "mask_ratio": mask_ratio,
        }
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            inference_dropout_removal_ratio=inference_dropout_removal_ratio,
            normalization_stats_path=normalization_stats_path,
            data_dir=data_dir,
        )

    def _load_model(self):
        from openmhc.models.lsm2.modules import LSM2Module  # lazy

        module = LSM2Module.load_from_checkpoint(
            str(self._ckpt_file),
            map_location=self._device,
        )
        return module.model

    def _inference_forward(self, x, inherited_mask):
        torch = self._torch
        model = self._model
        B, _, _ = x.shape

        x_patched = model.patch_embed(x)
        x_patched = x_patched + model.pos_embed.to(x.device, dtype=x.dtype)

        N = x_patched.shape[1]
        D = x_patched.shape[2]
        total_mask = inherited_mask

        ratio = (
            self._inference_dropout_removal_ratio
            if self._inference_dropout_removal_ratio is not None
            else model.dropout_removal_ratio
        )

        if ratio <= 0.0:
            # Deterministic fast path: keep every token, skip the unseeded
            # torch.rand + argsort shuffle entirely. Identity permutation.
            len_keep = N
            ids_restore = (
                torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
            )
            x_masked = x_patched
            kept_mask_status = total_mask
        else:
            len_keep = int(N * (1.0 - ratio))
            noise = torch.rand(B, N, device=x.device)
            priority = total_mask * 100.0 + noise
            ids_shuffle = torch.argsort(priority, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(
                x_patched, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )
            kept_mask_status = torch.gather(total_mask, dim=1, index=ids_keep)

        attn_mask = torch.zeros(B, 1, 1, len_keep, device=x.device)
        attn_mask.masked_fill_(
            kept_mask_status.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
        )

        latent = model.encoder(x_masked, attn_mask=attn_mask)
        pred = model.forward_decoder(latent, ids_restore, total_mask)
        return pred


# ---------------------------------------------------------------------------
# Weekly-sparse (per-day encoder + sparse cross-day decoder)
# ---------------------------------------------------------------------------


class LSM2WeeklySparseImputer(_LSM2ImputerBase):
    """LSM2 weekly-sparse imputer.

    Wraps :class:`openmhc.models.lsm2.WeeklySparseDecoderLSM2`. Splits a
    weekly tensor ``(B, 19, num_days * seq_length)`` into per-day slices,
    encodes each independently with the daily encoder, then reconstructs
    the full week through alternating day-local / cross-day decoder
    blocks (optionally calendar-aware via RoPE day embeddings).

    Args:
        model_path: Path to a Lightning ``.ckpt`` file or a directory.
        num_days, window_minutes, use_rope_day_embed, freeze_encoder
            (plus all the args from :class:`LSM2Imputer`): Architecture
            hyperparameters; see :class:`LSM2Imputer` for the loading caveat.
        device, inference_batch_size, inference_dropout_removal_ratio,
        normalization_stats_path, data_dir: See :class:`LSM2Imputer`.
    """

    model_name = "lsm2_weekly_sparse"

    def __init__(
        self,
        model_path: str | Path,
        version,
        *,
        # weekly-sparse-specific
        num_days: int = 7,
        window_minutes: int = 120,
        use_rope_day_embed: bool = True,
        freeze_encoder: bool = True,
        # shared with daily/weekly
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        device: str = "auto",
        inference_batch_size: int = 16,
        inference_dropout_removal_ratio: float | None = 0.0,
        normalization_stats_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        **_extra: Any,
    ) -> None:
        self._declared_arch = {
            "num_days": num_days,
            "window_minutes": window_minutes,
            "use_rope_day_embed": use_rope_day_embed,
            "freeze_encoder": freeze_encoder,
            "seq_length": seq_length,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
            "mlp_ratio": mlp_ratio,
            "mask_ratio": mask_ratio,
        }
        super().__init__(
            model_path,
            version=version,
            device=device,
            inference_batch_size=inference_batch_size,
            inference_dropout_removal_ratio=inference_dropout_removal_ratio,
            normalization_stats_path=normalization_stats_path,
            data_dir=data_dir,
        )

    def _load_model(self):
        from openmhc.models.lsm2.modules import WeeklySparseLSM2Module  # lazy

        module = WeeklySparseLSM2Module.load_from_checkpoint(
            str(self._ckpt_file),
            map_location=self._device,
        )
        return module.model

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        day_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Override to accept ``day_offsets`` for RoPE-aware decoding.

        The weekly-sparse model receives ``(B, 19, num_days * 1440)``
        windows from the harness (the evaluator's ``MultiDayImputationDataset``
        path, gated on ``cfg.data.n_days > 1``). When the
        ``WeeklySparseDecoderLSM2`` was trained with
        ``use_rope_day_embed=True`` (the d4 sparse decoder checkpoint), the
        cross-day attention blocks consume per-window calendar deltas to
        encode real-world day gaps inside each window. We forward
        ``day_offsets`` (``(B, num_days)`` int64, ``-1`` for padded slots)
        through to ``_inference_forward``; if the caller doesn't pass it,
        the model falls back to ``arange(num_days)`` internally — only
        correct when every window is calendar-consecutive.
        """
        torch = self._torch
        result = data.copy()
        N = data.shape[0]
        bs = max(1, self._inference_batch_size)

        valid = observed_mask * (1.0 - target_mask)

        for start in range(0, N, bs):
            end = min(start + bs, N)
            batch_data = data[start:end]
            batch_valid = valid[start:end]
            batch_target = target_mask[start:end]
            batch_offsets = (
                day_offsets[start:end] if day_offsets is not None else None
            )

            x_norm = self._normalize(batch_data)
            x_filled = np.where(np.isfinite(x_norm), x_norm, 0.0).astype(np.float32)

            x_t = torch.from_numpy(x_filled).to(self._device)
            valid_t = torch.from_numpy(batch_valid.astype(np.float32)).to(self._device)
            inherited_mask = self._create_inherited_mask(valid_t)

            with torch.no_grad():
                pred = self._inference_forward(
                    x_t, inherited_mask, day_offsets=batch_offsets
                )
                reconstructed = self._model.unpatchify(pred)
                if getattr(self._model, "use_hybrid_loss", False):
                    reconstructed[:, 7:, :] = torch.sigmoid(reconstructed[:, 7:, :])

            recon_np = reconstructed.detach().cpu().numpy()
            recon_np = self._denormalize(recon_np)

            tb = batch_target > 0.5
            result[start:end][tb] = recon_np[tb]

        return result.astype(np.float32, copy=False)

    def _inference_forward(self, x, inherited_mask, day_offsets=None):
        torch = self._torch
        model = self._model
        B = x.shape[0]
        D = model.num_days

        offsets_tensor = None
        if day_offsets is not None:
            offsets_tensor = torch.as_tensor(
                day_offsets, dtype=torch.long, device=self._device
            )

        # (B, C, D*L) → (B*D, C, L)
        x_days = x.reshape(B, model.in_channels, D, model.seq_length)
        x_days = (
            x_days.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B * D, model.in_channels, model.seq_length)
        )

        # Channel-major (B, C*D*T_per_day) → per-day (B*D, C*T_per_day).
        inh_days = (
            inherited_mask.reshape(
                B, model.in_channels, D, model.patches_per_channel_per_day
            )
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B * D, model.tokens_per_day)
        )

        x_tokens = model.patch_embed(x_days)
        x_tokens = x_tokens + model.pos_embed.to(x_tokens.device, dtype=x_tokens.dtype)

        BD, N, D_emb = x_tokens.shape
        total_mask = inh_days

        ratio = (
            self._inference_dropout_removal_ratio
            if self._inference_dropout_removal_ratio is not None
            else model.dropout_removal_ratio
        )

        if ratio <= 0.0:
            # Deterministic fast path: keep every token, skip the unseeded
            # torch.rand + argsort shuffle entirely. Identity permutation.
            len_keep = N
            ids_restore = (
                torch.arange(N, device=x_tokens.device).unsqueeze(0).expand(BD, N)
            )
            x_masked = x_tokens
            kept_mask_status = total_mask
        else:
            len_keep = int(N * (1.0 - ratio))
            noise = torch.rand(BD, N, device=x_tokens.device)
            priority = total_mask * 100.0 + noise
            ids_shuffle = torch.argsort(priority, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(
                x_tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D_emb)
            )
            kept_mask_status = torch.gather(total_mask, dim=1, index=ids_keep)

        attn_mask = torch.zeros(BD, 1, 1, len_keep, device=x_tokens.device)
        attn_mask.masked_fill_(
            kept_mask_status.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
        )

        latent = model.encoder(x_masked, attn_mask=attn_mask)

        # Fold per-day total_mask back to weekly channel-major.
        total_mask_week = (
            total_mask.reshape(B, D, model.in_channels, model.patches_per_channel_per_day)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, model.num_patches)
        )

        pred = model.forward_decoder(
            latent, ids_restore, total_mask_week, day_offsets=offsets_tensor
        )
        return pred
