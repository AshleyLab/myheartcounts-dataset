"""Weekly Sparse Decoder MAE: daily encoder + alternating day-local / cross-day decoder.

The encoder processes 7 daily slices independently at 10-minute patch resolution,
matching the pre-trained daily MAE architecture exactly. The decoder reconstructs
the full week using alternating day-local and cross-day sparse attention layers.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, PatchEmbed1D, ViTEncoder
from .positional import get_1d_sincos_pos_embed_from_grid

logger = logging.getLogger(__name__)


class WeeklySparseDecoderLSM2(nn.Module):
    """MAE with per-day encoder and sparse cross-day decoder for weekly data.

    Input shape: (B, in_channels, num_days * seq_length) e.g. (B, 19, 10080).
    Internally splits into 7 daily slices, encodes each independently, then
    decodes the full week with alternating day-local and cross-day attention.

    Args:
        seq_length: Per-day sequence length in minutes (default: 1440).
        patch_size: Patch size in minutes (default: 10).
        in_channels: Number of sensor channels (default: 19).
        embed_dim: Encoder embedding dimension.
        depth: Number of encoder transformer blocks.
        num_heads: Number of encoder attention heads.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Total decoder layers (alternates local/cross-day).
        decoder_num_heads: Number of decoder attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        qkv_bias: Whether to use bias in QKV projections.
        norm_pix_loss: Whether to normalize pixels before computing loss.
        dropout_removal_ratio: Fraction of tokens to physically remove in AIM.
        mask_ratio: Fraction of tokens to mask artificially.
        use_hybrid_loss: Use MSE for continuous + BCE for binary channels.
        continuous_channels: Channel indices for MSE loss.
        channel_weights: Optional per-channel loss weights.
        num_days: Number of days in the weekly window (default: 7).
        window_minutes: Cross-day attention window width in minutes (default: 120).
        use_rope_day_embed: Use calendar-aware RoPE day embeddings instead of
            fixed slot embeddings. Required for relaxed windowing where day
            offsets are not contiguous (default: False).
    """

    def __init__(  # noqa: D417
        self,
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
        qkv_bias: bool = True,
        norm_pix_loss: bool = False,
        dropout_removal_ratio: float = 0.5,
        mask_ratio: float = 0.5,
        use_hybrid_loss: bool = True,
        continuous_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        channel_weights: list[float] | None = None,
        num_days: int = 7,
        window_minutes: int = 120,
        use_rope_day_embed: bool = False,
    ):
        """Initialize WeeklySparseDecoderLSM2.

        Args:
            seq_length: Per-day sequence length in minutes.
            patch_size: Patch size in minutes.
            in_channels: Number of sensor channels.
            embed_dim: Encoder embedding dimension.
            depth: Encoder transformer depth.
            num_heads: Encoder attention heads.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Total decoder layers (alternates local/cross).
            decoder_num_heads: Decoder attention heads.
            mlp_ratio: MLP expansion ratio.
            qkv_bias: QKV bias.
            norm_pix_loss: Normalize pixels before loss.
            dropout_removal_ratio: Token dropout ratio in AIM.
            mask_ratio: Artificial masking ratio.
            use_hybrid_loss: MSE+BCE hybrid loss.
            continuous_channels: Channel indices for MSE loss.
            channel_weights: Per-channel loss weights.
            num_days: Days per weekly window.
            window_minutes: Cross-day attention window width in minutes.
            use_rope_day_embed: Use calendar-aware RoPE day embeddings instead
                of fixed slot embeddings. Required for relaxed windowing.
        """
        super().__init__()

        # Per-day dimensions (encoder operates at this scale)
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_removal_ratio = dropout_removal_ratio
        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        self.use_hybrid_loss = use_hybrid_loss
        self.continuous_channels = set(continuous_channels)
        self.channel_weights = channel_weights
        self.num_days = num_days
        self.use_rope_day_embed = use_rope_day_embed

        if channel_weights is not None and len(channel_weights) != in_channels:
            raise ValueError(
                f"channel_weights must have length {in_channels}, got {len(channel_weights)}"
            )

        # Daily token dimensions
        self.patches_per_channel_per_day = seq_length // patch_size  # 144
        self.tokens_per_day = in_channels * self.patches_per_channel_per_day  # 2736

        # Weekly token dimensions (used by loss, patchify, unpatchify)
        self.num_patches_per_channel = self.patches_per_channel_per_day * num_days  # 1008
        self.num_patches = self.tokens_per_day * num_days  # 19152

        # Cross-day windowing
        self.patches_per_window = window_minutes // patch_size  # 12
        self.num_windows = self.patches_per_channel_per_day // self.patches_per_window  # 12
        if self.patches_per_channel_per_day % self.patches_per_window != 0:
            raise ValueError(
                f"patches_per_channel_per_day ({self.patches_per_channel_per_day}) must be "
                f"divisible by patches_per_window ({self.patches_per_window}). "
                f"Adjust window_minutes ({window_minutes}) or patch_size ({patch_size})."
            )
        self.tokens_per_cross_window = (
            num_days * in_channels * self.patches_per_window
        )  # 1596

        # --- Encoder (matches daily MAE architecture) ---
        self.patch_embed = PatchEmbed1D(seq_length, patch_size, in_channels, embed_dim)
        self.pos_embed = self._build_daily_2d_sincos_pos_embed(embed_dim)
        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        # --- Encoder-to-decoder bridge ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

        # --- Decoder positional embeddings ---
        # Daily sincos (tiled per day at runtime) + day identity (learned or RoPE)
        self.decoder_pos_embed = self._build_daily_2d_sincos_pos_embed(decoder_embed_dim)
        if not use_rope_day_embed:
            self.day_embed = nn.Parameter(torch.zeros(1, num_days, 1, decoder_embed_dim))
        else:
            self.day_embed = None
            from .positional import RotaryDayEmbedding

            head_dim = decoder_embed_dim // decoder_num_heads
            self.rope_day = RotaryDayEmbedding(head_dim=head_dim)

        # --- Sparse decoder: alternating local / cross-day blocks ---
        self.decoder_local_blocks = nn.ModuleList()
        self.decoder_cross_blocks = nn.ModuleList()
        for i in range(decoder_depth):
            if i % 2 == 0:
                self.decoder_local_blocks.append(
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias)
                )
            else:
                self.decoder_cross_blocks.append(
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=qkv_bias)
                )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)

        self._init_weights()
        self._build_channel_masks()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if self.day_embed is not None:
            nn.init.trunc_normal_(self.day_embed, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_daily_2d_sincos_pos_embed(self, embed_dim: int) -> torch.Tensor:
        """Build 2D sincos positional embedding for one day (channel x time)."""
        channel_positions = []
        time_positions = []
        for ch in range(self.in_channels):
            for t in range(self.patches_per_channel_per_day):
                channel_positions.append(ch)
                time_positions.append(t)

        ch_arr = np.array(channel_positions, dtype=np.float32)
        t_arr = np.array(time_positions, dtype=np.float32)

        ch_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, ch_arr)
        t_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, t_arr)

        pos_embed = np.concatenate([ch_embed, t_embed], axis=1)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)  # (1, tokens_per_day, D)

    def _build_channel_masks(self) -> None:
        """Pre-compute weekly channel masks for hybrid loss."""
        continuous_mask = []
        patch_weights = []

        for ch in range(self.in_channels):
            is_continuous = ch in self.continuous_channels
            w = self.channel_weights[ch] if self.channel_weights is not None else 1.0
            for _ in range(self.num_patches_per_channel):
                continuous_mask.append(is_continuous)
                patch_weights.append(w)

        self.register_buffer(
            "continuous_patch_mask",
            torch.tensor(continuous_mask, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "patch_weights",
            torch.tensor(patch_weights, dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Masking (operates on per-day tokens)
    # ------------------------------------------------------------------

    def generate_artificial_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate per-day artificial mask (random / temporal / sensor strategies).

        Args:
            x: (B_days, tokens_per_day, D) where B_days = B * num_days.

        Returns:
            Mask of shape (B_days, tokens_per_day), 1 = masked.
        """
        B, N, _ = x.shape
        device = x.device
        T = self.patches_per_channel_per_day
        C = self.in_channels

        strategies = torch.randint(0, 3, (B,), device=device)

        # Random
        random_mask = (torch.rand(B, N, device=device) < self.mask_ratio).float()

        # Temporal slice
        temp_mask_t = (torch.rand(B, T, device=device) < 0.5).float()
        temp_mask = temp_mask_t.repeat(1, C)

        # Sensor slice
        sensor_mask_c = (torch.rand(B, C, device=device) < 0.5).float()
        sensor_mask = sensor_mask_c.repeat_interleave(T, dim=1)

        is_random = (strategies == 0).unsqueeze(1)
        is_temp = (strategies == 1).unsqueeze(1)
        is_sensor = (strategies == 2).unsqueeze(1)

        return random_mask * is_random + temp_mask * is_temp + sensor_mask * is_sensor

    def aim_masking(
        self, x: torch.Tensor, inherited_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adaptive and Inherited Masking on per-day tokens.

        Args:
            x: (B_days, tokens_per_day, embed_dim)
            inherited_mask: (B_days, tokens_per_day)

        Returns:
            x_masked: (B_days, len_keep, embed_dim)
            total_mask: (B_days, tokens_per_day)
            artificial_mask: (B_days, tokens_per_day)
            ids_restore: (B_days, tokens_per_day)
            attn_mask: (B_days, 1, 1, len_keep)
        """
        B, N, D = x.shape

        artificial_mask = self.generate_artificial_mask(x)
        total_mask = (inherited_mask.bool() | artificial_mask.bool()).float()

        len_keep = int(N * (1 - self.dropout_removal_ratio))

        noise = torch.rand(B, N, device=x.device)
        priority = total_mask * 100.0 + noise

        ids_shuffle = torch.argsort(priority, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        kept_mask_status = torch.gather(total_mask, dim=1, index=ids_keep)
        attn_mask = torch.zeros(B, 1, 1, len_keep, device=x.device)
        attn_mask.masked_fill_(kept_mask_status.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        return x_masked, total_mask, artificial_mask, ids_restore, attn_mask

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def forward_encoder(
        self, x_week: torch.Tensor, inherited_mask_week: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode weekly input by splitting into daily slices.

        Args:
            x_week: (B, C, num_days * seq_length) e.g. (B, 19, 10080)
            inherited_mask_week: (B, num_patches) e.g. (B, 19152)

        Returns:
            latent: (B*num_days, len_keep, embed_dim)
            total_mask_week: (B, num_patches)
            artificial_mask_week: (B, num_patches)
            ids_restore: (B*num_days, tokens_per_day)
        """
        B = x_week.shape[0]
        D_days = self.num_days

        # Split into daily slices
        # (B, C, D*L) -> (B, C, D, L) -> (B*D, C, L)
        x_days = x_week.reshape(B, self.in_channels, D_days, self.seq_length)
        x_days = x_days.permute(0, 2, 1, 3).contiguous().reshape(B * D_days, self.in_channels, self.seq_length)

        # Split inherited mask: channel-major (B, C*D*T_per_day) -> per-day
        # channel-major (B*D, C*T_per_day).
        # create_inherited_mask lays out the mask as [ch0 * 1008, ch1 * 1008, ...]
        # so we must permute channels past days before flattening — a bare
        # .reshape(B, D, N_day) would mix channels across days.
        inh_days = (
            inherited_mask_week
            .reshape(B, self.in_channels, D_days, self.patches_per_channel_per_day)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B * D_days, self.tokens_per_day)
        )

        # Patch embed + positional encoding
        x_tokens = self.patch_embed(x_days)  # (B*D, tokens_per_day, embed_dim)
        x_tokens = x_tokens + self.pos_embed.to(x_tokens.device, dtype=x_tokens.dtype)

        # AIM masking (each day is an independent sample in the batch)
        latent, total_mask, artificial_mask, ids_restore, attn_mask = self.aim_masking(
            x_tokens, inh_days
        )

        # Encoder
        latent = self.encoder(latent, attn_mask=attn_mask)

        # Convert per-day (B*D, C*T_per_day) masks to weekly channel-major
        # (B, C*D*T_per_day) so the caller (forward / forward_loss) can align
        # them with patchify(x), continuous_patch_mask, patch_weights, and the
        # HR slice — all of which assume channel-major layout.
        total_mask_week = (
            total_mask
            .reshape(B, D_days, self.in_channels, self.patches_per_channel_per_day)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, self.num_patches)
        )
        artificial_mask_week = (
            artificial_mask
            .reshape(B, D_days, self.in_channels, self.patches_per_channel_per_day)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, self.num_patches)
        )

        return latent, total_mask_week, artificial_mask_week, ids_restore

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        total_mask_week: torch.Tensor,
        day_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse weekly decoder with alternating day-local and cross-day layers.

        Args:
            latent: (B*num_days, len_keep, embed_dim)
            ids_restore: (B*num_days, tokens_per_day)
            total_mask_week: (B, num_patches)
            day_offsets: (B, num_days) calendar day offsets for RoPE, or None.

        Returns:
            pred: (B, num_patches, patch_size)
        """
        BD = latent.shape[0]
        B = BD // self.num_days
        D_days = self.num_days
        N_day = self.tokens_per_day

        # Project encoder output to decoder dim
        x = self.encoder_to_decoder(latent)
        _, L_keep, D_dec = x.shape
        L_full = ids_restore.shape[1]  # tokens_per_day

        # Unshuffle + mask token fill (per day)
        mask_tokens_drop = self.mask_token.expand(BD, L_full - L_keep, -1)
        x_ = torch.cat([x, mask_tokens_drop], dim=1)
        x_full = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        )

        # Replace masked positions with learnable mask token.
        # total_mask_week is channel-major (B, C*D*T_per_day); the decoder's
        # internal layout here is per-day (B*D, N_day), so permute back.
        total_mask_days = (
            total_mask_week
            .reshape(B, self.in_channels, D_days, self.patches_per_channel_per_day)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(BD, N_day)
        )
        mask_bool = total_mask_days.unsqueeze(-1).bool()
        full_mask_tokens = self.mask_token.expand(BD, L_full, -1)
        x_full = torch.where(mask_bool, full_mask_tokens, x_full)

        # Add daily sincos positional embedding (same for each day)
        x_full = x_full + self.decoder_pos_embed.to(x_full.device, dtype=x_full.dtype)

        # Reshape to (B, D, tokens_per_day, D_dec) and add day embedding
        x_week = x_full.reshape(B, D_days, N_day, D_dec)
        if self.day_embed is not None:
            x_week = x_week + self.day_embed.to(x_week.device, dtype=x_week.dtype)

        # Alternating sparse decoder layers
        local_idx = 0
        cross_idx = 0
        total_layers = len(self.decoder_local_blocks) + len(self.decoder_cross_blocks)

        for layer_i in range(total_layers):
            if layer_i % 2 == 0:
                # Day-local attention: (B*D, tokens_per_day, D_dec)
                x_flat = x_week.reshape(B * D_days, N_day, D_dec)
                x_flat = self.decoder_local_blocks[local_idx](x_flat)
                x_week = x_flat.reshape(B, D_days, N_day, D_dec)
                local_idx += 1
            else:
                # Cross-day window attention
                x_week = self._cross_day_attention(
                    x_week, self.decoder_cross_blocks[cross_idx],
                    day_offsets=day_offsets,
                )
                cross_idx += 1

        # Final norm + prediction.
        # x_week is (B, D, N_day=C*T_per_day, D_dec) with channel-major
        # ordering within each day. The target (patchify) is channel-major at
        # the weekly level: [ch0 * D*T_per_day, ch1 * D*T_per_day, ...]. Permute
        # day past channel and flatten so pred matches the target's token order
        # (and continuous_patch_mask / patch_weights / HR slice).
        x_week = x_week.reshape(
            B, D_days, self.in_channels, self.patches_per_channel_per_day, D_dec
        )
        x_week = x_week.permute(0, 2, 1, 3, 4).contiguous()
        x_out = x_week.reshape(B, self.num_patches, D_dec)
        x_out = self.decoder_norm(x_out)
        pred = self.decoder_pred(x_out)
        return pred

    def _cross_day_attention(
        self,
        x: torch.Tensor,
        block: Block,
        day_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Regroup tokens by time window across days and apply attention.

        Args:
            x: (B, num_days, tokens_per_day, D)
            block: Transformer block to apply within each window.
            day_offsets: (B, num_days) calendar day offsets for RoPE, or None.

        Returns:
            (B, num_days, tokens_per_day, D)
        """
        B, D_days, N_day, D_dec = x.shape
        C = self.in_channels
        T_day = self.patches_per_channel_per_day  # 144
        W = self.num_windows  # 12
        Pw = self.patches_per_window  # 12

        # (B, D, C*T, D_dec) -> (B, D, C, T, D_dec)
        x = x.reshape(B, D_days, C, T_day, D_dec)
        # (B, D, C, W, Pw, D_dec)
        x = x.reshape(B, D_days, C, W, Pw, D_dec)
        # Group by window: (B, W, D, C, Pw, D_dec)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        # Flatten window tokens: (B*W, D*C*Pw, D_dec)
        x = x.reshape(B * W, D_days * C * Pw, D_dec)

        rope_fn = None
        if self.use_rope_day_embed:
            if day_offsets is not None:
                max_pos = self.rope_day.cos_table.shape[0] - 1
                clamped = day_offsets.clamp(min=0, max=max_pos)
                # (B, D) → (B, D*C*Pw) → (B, W, D*C*Pw) → (B*W, D*C*Pw)
                day_positions = clamped.repeat_interleave(C * Pw, dim=1)
                day_positions = (
                    day_positions.unsqueeze(1).expand(-1, W, -1).reshape(B * W, -1)
                )
            else:
                day_positions = torch.arange(
                    D_days, device=x.device
                ).repeat_interleave(C * Pw)
            rope_fn = self.rope_day.get_rope_fn(day_positions)
        x = block(x, rope_fn=rope_fn)

        # Reverse: (B*W, D*C*Pw, D_dec) -> (B, W, D, C, Pw, D_dec)
        x = x.reshape(B, W, D_days, C, Pw, D_dec)
        # (B, D, C, W, Pw, D_dec)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, D, C*T, D_dec) = (B, D, tokens_per_day, D_dec)
        x = x.reshape(B, D_days, N_day, D_dec)
        return x

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reconstruction loss on masked patches (weekly).

        Args:
            imgs: (B, C, num_days * seq_length)
            pred: (B, num_patches, patch_size)
            mask: (B, num_patches) — 1 where loss should be computed
            return_per_sample: Return per-sample continuous/binary losses.
        """
        target = self.patchify(imgs)
        B, N, P = target.shape

        # HR per-minute mask (channel 5)
        hr_start = 5 * self.num_patches_per_channel
        hr_end = 6 * self.num_patches_per_channel
        hr_minute_mask = torch.ones(B, N, P, device=target.device, dtype=target.dtype)
        hr_minute_mask[:, hr_start:hr_end, :] = (target[:, hr_start:hr_end, :] != 0).float()

        if not self.use_hybrid_loss:
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            loss = (pred - target) ** 2
            valid_counts = hr_minute_mask.sum(dim=-1).clamp(min=1)
            loss = (loss * hr_minute_mask).sum(dim=-1) / valid_counts
        else:
            pred_f32 = pred.float()
            target_f32 = target.float()
            continuous_mask = self.continuous_patch_mask

            if self.norm_pix_loss:
                cont_target = target_f32[:, continuous_mask, :]
                mean = cont_target.mean(dim=-1, keepdim=True)
                var = cont_target.var(dim=-1, keepdim=True)
                cont_target_norm = (cont_target - mean) / (var + 1.0e-6) ** 0.5
                target_f32 = target_f32.clone()
                target_f32[:, continuous_mask, :] = cont_target_norm

            mse_loss = (pred_f32 - target_f32) ** 2
            hr_minute_mask_f32 = hr_minute_mask.float()
            valid_counts = hr_minute_mask_f32.sum(dim=-1).clamp(min=1)
            mse_per_patch = (mse_loss * hr_minute_mask_f32).sum(dim=-1) / valid_counts

            binary_mask = ~continuous_mask
            pred_binary = pred_f32[:, binary_mask, :]
            target_binary = target_f32[:, binary_mask, :].clamp(0.0, 1.0)

            bce_per_patch = F.binary_cross_entropy_with_logits(
                pred_binary, target_binary, reduction="none"
            ).mean(dim=-1)

            loss = torch.zeros(B, N, device=pred.device, dtype=torch.float32)
            loss[:, continuous_mask] = mse_per_patch[:, continuous_mask]
            loss[:, binary_mask] = bce_per_patch

            loss = loss * self.patch_weights

        per_sample_losses = None
        if return_per_sample:
            cont_mask_vals = mask * self.continuous_patch_mask.float()
            per_sample_cont = (loss * cont_mask_vals).sum(dim=1) / cont_mask_vals.sum(
                dim=1
            ).clamp(min=1)

            bin_mask_vals = mask * (~self.continuous_patch_mask).float()
            per_sample_bin = (loss * bin_mask_vals).sum(dim=1) / bin_mask_vals.sum(dim=1).clamp(
                min=1
            )

            per_sample_losses = {
                "continuous_loss": per_sample_cont,
                "binary_loss": per_sample_bin,
            }

        if mask.sum() == 0:
            scalar_loss = loss.mean()
        else:
            scalar_loss = (loss * mask).sum() / mask.sum()

        if return_per_sample:
            return scalar_loss, per_sample_losses
        return scalar_loss

    # ------------------------------------------------------------------
    # Patchify / unpatchify (weekly)
    # ------------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, num_days*seq_length) -> (B, num_patches, patch_size)."""
        p = self.patch_size
        B, C, L = imgs.shape
        x = imgs.reshape(B, C, L // p, p)
        return x.contiguous().view(B, -1, p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_size) -> (B, C, num_days*seq_length)."""
        p = self.patch_size
        h = self.in_channels
        w = self.num_patches_per_channel
        B = x.shape[0]
        x = x.view(B, h, w, p)
        return x.contiguous().view(B, h, w * p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        inherited_mask: torch.Tensor | None = None,
        return_per_sample: bool = False,
        day_offsets: torch.Tensor | None = None,
        original_target: torch.Tensor | None = None,
        day_recon_patch_mask: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            dict[str, torch.Tensor],
            torch.Tensor | None,
        ]
    ):
        """Full forward pass: encode daily, decode weekly, compute loss.

        Args:
            x: (B, C, num_days * seq_length) with NaN replaced by 0.
            inherited_mask: (B, num_patches) or None.
            return_per_sample: Return per-sample loss breakdown.
            day_offsets: (B, num_days) calendar day offsets for RoPE, or None.
            original_target: (B, C, num_days * seq_length) pre-day-masking target, or None.
            day_recon_patch_mask: (B, num_patches) 1 for artificially day-masked
                patches that had real (non-NaN) ground truth. The caller is
                responsible for pre-gating against the natural inherited mask
                (see ``WeeklySparseMAEModule._shared_step``); this forward does
                not gate again.

        Returns:
            (loss, pred, total_mask, day_recon_loss) or
            (loss, pred, total_mask, per_sample_losses, day_recon_loss).
            ``day_recon_loss`` is the loss restricted to day-masked patches that
            had a real ground-truth target (non-inherited), or ``None`` when
            day reconstruction is inactive.
        """
        if inherited_mask is None:
            B = x.shape[0]
            inherited_mask = torch.zeros((B, self.num_patches), device=x.device)

        latent, total_mask, artificial_mask, ids_restore = self.forward_encoder(
            x, inherited_mask
        )
        pred = self.forward_decoder(latent, ids_restore, total_mask, day_offsets=day_offsets)

        loss_mask = artificial_mask * (1 - inherited_mask)

        # day_recon_patch_mask is pre-gated by the caller (only set for
        # day-masked positions that had real ground truth), so we add it
        # in directly. Re-gating here against ``inherited_mask`` would zero
        # it out, since day-masked positions are themselves part of the
        # encoder-input ``inherited_mask``.
        day_recon_loss_mask = None
        if day_recon_patch_mask is not None:
            day_recon_loss_mask = day_recon_patch_mask
            loss_mask = torch.clamp(loss_mask + day_recon_loss_mask, max=1.0)

        # Use original target for loss when day reconstruction is active
        loss_target = original_target if original_target is not None else x
        loss_result = self.forward_loss(
            loss_target, pred, loss_mask, return_per_sample=return_per_sample
        )

        # Optional day-recon-only scalar for diagnostics; None when day-recon
        # is inactive or the gated mask is empty.
        day_recon_loss: torch.Tensor | None = None
        if day_recon_loss_mask is not None and day_recon_loss_mask.sum() > 0:
            day_recon_loss = self.forward_loss(loss_target, pred, day_recon_loss_mask)

        if return_per_sample:
            loss, per_sample_losses = loss_result
            return loss, pred, total_mask, per_sample_losses, day_recon_loss
        return loss_result, pred, total_mask, day_recon_loss

    # ------------------------------------------------------------------
    # Weight loading from daily checkpoint
    # ------------------------------------------------------------------

    def load_daily_encoder_weights(self, checkpoint_path: str) -> list[str]:
        """Load encoder weights from a pre-trained daily MAE checkpoint.

        Copies: patch_embed, encoder, pos_embed, encoder_to_decoder,
        mask_token, decoder_pred, decoder_pos_embed.

        Args:
            checkpoint_path: Path to a MAEModule checkpoint (.ckpt).

        Returns:
            List of top-level component names that were loaded.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # MAEModule stores weights under "model." prefix
        prefix = "model."
        daily_state = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }
        if not daily_state:
            daily_state = state

        loaded = []

        # Direct copies (identical architecture)
        for name in [
            "patch_embed",
            "encoder",
            "encoder_to_decoder",
            "decoder_pred",
        ]:
            sub_state = {
                k[len(name) + 1 :]: v
                for k, v in daily_state.items()
                if k.startswith(name + ".")
            }
            if sub_state:
                getattr(self, name).load_state_dict(sub_state)
                loaded.append(name)

        # Parameters (pos_embed/decoder_pos_embed are deterministic sinusoidal,
        # computed identically by both models — no need to transfer)
        if "mask_token" in daily_state:
            self.mask_token.data.copy_(daily_state["mask_token"])
            loaded.append("mask_token")

        logger.info("Loaded daily encoder weights: %s", loaded)
        return loaded
