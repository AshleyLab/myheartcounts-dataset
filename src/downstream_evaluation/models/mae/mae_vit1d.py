"""Masked Autoencoder Vision Transformer for 1D data with LSM-2/AIM."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import PatchEmbed1D, ViTEncoder
from .positional import get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderViT1D_LSM2(nn.Module):
    """Masked Autoencoder Vision Transformer for 1D data with Adaptive and Inherited Masking.

    This implements the LSM-2 / AIM approach for handling data with both artificial
    (self-supervised) masking and inherited (real-world missing data) masking.

    Args:
        seq_length: Length of input sequence (default: 1440 minutes per day)
        patch_size: Size of each patch (default: 10 minutes)
        in_channels: Number of input channels (default: 19 health metrics)
        embed_dim: Embedding dimension for encoder
        depth: Number of transformer blocks in encoder
        num_heads: Number of attention heads in encoder
        decoder_embed_dim: Embedding dimension for decoder
        decoder_depth: Number of transformer blocks in decoder
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: MLP hidden dimension ratio
        norm_pix_loss: Whether to normalize pixels before computing loss
        dropout_removal_ratio: Fraction of tokens to physically remove (D)
        mask_ratio: Fraction of tokens to mask artificially (default: 0.5)
        use_hybrid_loss: Whether to use hybrid MSE+BCE loss (MSE for continuous,
            BCE for binary channels). Default: False (use MSE for all).
        continuous_channels: Tuple of channel indices to use MSE loss for.
            Default: (0, 1, 2, 3, 4, 5, 6) for continuous metrics.
        channel_weights: Optional list of per-channel weights for the loss.
            Must have length equal to in_channels. Default: None (uniform weights).
    """

    def __init__(
        self,
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_pix_loss: bool = False,
        dropout_removal_ratio: float = 0.5,
        mask_ratio: float = 0.8,
        use_hybrid_loss: bool = False,
        continuous_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        channel_weights: list[float] | None = None,
    ):
        """Initialize MAE ViT 1D model.

        Args:
            seq_length: Sequence length.
            patch_size: Patch size.
            in_channels: Input channels.
            embed_dim: Embedding dimension.
            depth: Encoder depth.
            num_heads: Number of attention heads.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Decoder depth.
            decoder_num_heads: Decoder num heads.
            mlp_ratio: MLP ratio.
            qkv_bias: Whether to use bias term in QKV projections.
            norm_pix_loss: Whether to normalize pixel loss.
            dropout_removal_ratio: Dropout removal ratio.
            mask_ratio: Masking ratio.
            use_hybrid_loss: Whether to use hybrid loss.
            continuous_channels: Continuous channel indices.
            channel_weights: Per-channel weights.
        """
        super().__init__()
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

        # Validate channel_weights if provided
        if channel_weights is not None and len(channel_weights) != in_channels:
            raise ValueError(
                f"channel_weights must have length {in_channels}, got {len(channel_weights)}"
            )

        # Patch Embed
        self.patch_embed = PatchEmbed1D(seq_length, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.total_patches
        self.num_patches_per_channel = self.patch_embed.num_patches_per_channel

        # Positional Embeddings (2D: Channel x Time)
        self.pos_embed = self._build_2d_sincos_pos_embed(embed_dim)
        self.decoder_pos_embed = self._build_2d_sincos_pos_embed(decoder_embed_dim)

        # Encoder
        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder = ViTEncoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)

        self._init_weights()

        # Pre-compute channel masks (registered as buffers for device handling)
        # Always built: needed for hybrid loss and per-sample residual extraction
        self._build_channel_masks()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_channel_masks(self) -> None:
        """Pre-compute channel masks for hybrid loss.

        Creates:
            - continuous_patch_mask: Boolean tensor (N,) where True = continuous channel
            - patch_weights: Float tensor (N,) with per-patch weights from channel_weights
        """
        # Build mask: patches are ordered by channel then time
        # i.e., patches [0, num_patches_per_channel) belong to channel 0, etc.
        continuous_mask = []
        patch_weights = []

        for channel_idx in range(self.in_channels):
            is_continuous = channel_idx in self.continuous_channels
            weight = self.channel_weights[channel_idx] if self.channel_weights is not None else 1.0

            for _ in range(self.num_patches_per_channel):
                continuous_mask.append(is_continuous)
                patch_weights.append(weight)

        # Register as buffers so they move with the model to GPU
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

    def _build_2d_sincos_pos_embed(self, embed_dim: int) -> torch.Tensor:
        """Generate 2D positional embedding (Channel ID + Time Index)."""
        channel_positions = []
        time_positions = []
        for channel_idx in range(self.in_channels):
            for time_idx in range(self.num_patches_per_channel):
                channel_positions.append(channel_idx)
                time_positions.append(time_idx)

        channel_positions = np.array(channel_positions, dtype=np.float32)
        time_positions = np.array(time_positions, dtype=np.float32)

        channel_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, channel_positions)
        time_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, time_positions)

        pos_embed = np.concatenate([channel_embed, time_embed], axis=1)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def generate_artificial_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate artificial mask using a mix of strategies: Random, Temporal, or Sensor.

        According to LSM-2 paper:
        - 33% chance: Random Imputation (default 80% masking)
        - 33% chance: Temporal Slice (default 50% masking)
        - 33% chance: Sensor Slice (default 50% masking)

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Mask tensor of shape (B, N) where 1 indicates artificial mask
        """
        B, N, _ = x.shape
        device = x.device

        # Determine strategy for each sample in the batch
        # 0: Random, 1: Temporal, 2: Sensor
        strategies = torch.randint(0, 3, (B,), device=device)

        final_mask = torch.zeros(B, N, device=device)

        # 1. Random Imputation Masking
        # Mask individual patches with probability self.mask_ratio (e.g., 0.8)
        random_noise = torch.rand(B, N, device=device)
        random_mask = (random_noise < self.mask_ratio).float()

        # 2. Temporal Slice Masking
        # Mask all channels for specific time steps
        # Reshape to (B, C, T) to easily mask time steps
        # T = num_patches_per_channel, C = in_channels
        T = self.num_patches_per_channel
        C = self.in_channels

        # Generate mask for time steps (B, T)
        # Use a separate ratio for slices if desired, typically 0.5 in paper
        temp_ratio = 0.5
        temp_noise = torch.rand(B, T, device=device)
        temp_mask_t = (temp_noise < temp_ratio).float()  # (B, T)

        # Expand to all channels: (B, T) -> (B, C, T) -> Flatten to (B, N)
        # Note: Our flattening is Channel-First (C, T), so we repeat for C
        # But wait, our patches are flattened as [Ch0_t0...Ch0_tN, Ch1_t0...Ch1_tN]
        # So we need to tile the time mask C times.
        temp_mask = temp_mask_t.repeat(1, C)  # (B, C*T) = (B, N)

        # 3. Sensor Slice Masking
        # Mask all time steps for specific channels
        sensor_ratio = 0.5
        sensor_noise = torch.rand(B, C, device=device)
        sensor_mask_c = (sensor_noise < sensor_ratio).float()  # (B, C)

        # Expand to all time steps: (B, C) -> (B, C, T) -> Flatten
        # We need each channel bit repeated T times contiguously
        sensor_mask = sensor_mask_c.repeat_interleave(T, dim=1)  # (B, C*T) = (B, N)

        # Apply the selected strategy for each batch element
        # We can use broadcasting to select rows
        is_random = (strategies == 0).unsqueeze(1)
        is_temp = (strategies == 1).unsqueeze(1)
        is_sensor = (strategies == 2).unsqueeze(1)

        final_mask = random_mask * is_random + temp_mask * is_temp + sensor_mask * is_sensor

        return final_mask

    def aim_masking(
        self, x: torch.Tensor, inherited_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adaptive and Inherited Masking (AIM).

        Returns:
            x_masked: Masked tokens of shape (B, len_keep, D)
            total_mask: Combined mask of shape (B, N)
            artificial_mask: Artificial mask only of shape (B, N)
            ids_restore: Indices to restore original order of shape (B, N)
            attn_mask: Attention mask for encoder of shape (B, 1, 1, len_keep)
        """
        B, N, D = x.shape

        # 1. Artificial Mask
        artificial_mask = self.generate_artificial_mask(x)

        # 2. Total Union Mask
        total_mask = (inherited_mask.bool() | artificial_mask.bool()).float()

        # 3. Prioritized Dropout Removal
        # We physically remove 'len_drop' tokens.
        # Priority: Keep Observed (0), Drop Missing (1).
        len_keep = int(N * (1 - self.dropout_removal_ratio))

        # Add noise to break ties.
        # Value ~0 for observed, Value ~100 for missing.
        # Sorting (Low->High) puts Observed first (keep), Missing last (drop).
        noise = torch.rand(B, N, device=x.device)
        priority = total_mask * 100.0 + noise

        ids_shuffle = torch.argsort(priority, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the ones with lowest priority (Observed + some Missing overflow)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 4. Attention Mask for Encoder
        # Identify which of the kept tokens are actually missing
        kept_mask_status = torch.gather(total_mask, dim=1, index=ids_keep)

        # Create additive mask: 0 for valid, -inf for missing
        attn_mask = torch.zeros(B, 1, 1, len_keep, device=x.device)
        # Broadcast masked positions to columns (keys)
        # (B, 1, 1, N_keep) applied to (B, H, N_keep, N_keep)
        # If position j is masked, it cannot be attended to.
        attn_mask.masked_fill_(kept_mask_status.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        return x_masked, total_mask, artificial_mask, ids_restore, attn_mask

    def forward_encoder(
        self, x: torch.Tensor, inherited_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Returns latent, total_mask, artificial_mask, ids_restore.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device, dtype=x.dtype)

        latent, total_mask, artificial_mask, ids_restore, attn_mask = self.aim_masking(
            x, inherited_mask
        )
        latent = self.encoder(latent, attn_mask=attn_mask)

        return latent, total_mask, artificial_mask, ids_restore

    def forward_features_full(
        self,
        x: torch.Tensor,
        inherited_mask: torch.Tensor | None = None,
        use_decoder: bool = True,
    ) -> torch.Tensor:
        """Runs the model to produce a dense, full-length feature sequence.

        Args:
            x: Input (B, C, L)
            inherited_mask: (B, N) missingness mask
            use_decoder: If True, runs the lightweight decoder blocks to refine
                         the mask tokens before output. Recommended for segmentation.

        Returns:
            Full sequence features of shape (B, num_patches, decoder_embed_dim)
        """
        # 1. ENCODER (Sparse)
        # -------------------
        # Only process observed tokens. latent shape: (B, len_keep, embed_dim)
        latent, total_mask, artificial_mask, ids_restore = self.forward_encoder(x, inherited_mask)

        # 2. BRIDGE (Fill Gaps)
        # ---------------------
        # Project Encoder Dim -> Decoder Dim
        x_mapped = self.encoder_to_decoder(latent)

        B, len_keep, D_dec = x_mapped.shape
        L_full = ids_restore.shape[1]

        # Create mask tokens for ALL missing positions (both artificial and inherited)
        # total_mask tells us exactly which positions were dropped/missing.
        # Note: In standard MAE, we append tokens and unshuffle.
        # Here we follow the logic in forward_decoder to ensure rigorous reconstruction.

        # A. Append placeholders for dropped tokens to reach full length
        # (This matches the standard MAE unshuffle logic)
        mask_tokens_append = self.mask_token.repeat(B, L_full - len_keep, 1)
        x_concat = torch.cat([x_mapped, mask_tokens_append], dim=1)

        # B. Unshuffle to original order
        x_full = torch.gather(x_concat, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        # C. Explicitly overwrite Missing/Masked positions with the learnable mask_token
        # This ensures that even if a token 'leaked' through, if it was masked,
        # it is replaced by the learnable concept of "Missing".
        mask_bool = total_mask.unsqueeze(-1).bool()  # (B, N, 1)
        full_mask_tokens = self.mask_token.repeat(B, L_full, 1)
        x_full = torch.where(mask_bool, full_mask_tokens, x_full)

        # D. Add Decoder Positional Embeddings
        # Crucial: This tells the model "This mask token is at Minute 100" vs "Minute 500"
        x_full = x_full + self.decoder_pos_embed.to(x.device, dtype=x.dtype)

        # 3. DECODER (Refinement)
        # -----------------------
        # If we use the decoder, the mask tokens interact with observed tokens
        # to infer context.
        if use_decoder:
            x_full = self.decoder(x_full)

        # Output shape: (B, num_patches, decoder_embed_dim)
        return x_full

    def forward_encoder_features(
        self, x: torch.Tensor, inherited_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract encoder features for fine-tuning (no decoder).

        Args:
            x: Input tensor of shape (B, C, L) e.g. (B, 19, 1440)
            inherited_mask: Optional missing data mask of shape (B, N).
                           If None, assumes no missing data (all 0).

        Returns:
            latent: Encoded representations of shape (B, len_keep, embed_dim)
            ids_keep: Indices of kept tokens of shape (B, len_keep)
            channel_ids: Channel ID for each kept token of shape (B, len_keep)
        """
        B, _, L = x.shape

        if inherited_mask is None:
            inherited_mask = torch.zeros((B, self.num_patches), device=x.device)

        # Patchify and add positional embeddings
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device, dtype=x.dtype)

        # Apply AIM Masking (same logic as aim_masking but we also need ids_keep)
        N = x.shape[1]
        artificial_mask = self.generate_artificial_mask(x)
        total_mask = (inherited_mask.bool() | artificial_mask.bool()).float()

        len_keep = int(N * (1 - self.dropout_removal_ratio))

        noise = torch.rand(B, N, device=x.device)
        priority = total_mask * 100.0 + noise

        ids_shuffle = torch.argsort(priority, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather kept tokens
        latent = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # Attention mask for encoder
        kept_mask_status = torch.gather(total_mask, dim=1, index=ids_keep)
        attn_mask = torch.zeros(B, 1, 1, len_keep, device=x.device)
        attn_mask.masked_fill_(kept_mask_status.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        # Encoder pass
        latent = self.encoder(latent, attn_mask=attn_mask)

        # Compute channel IDs for each kept token
        # Patches are ordered: [ch0_p0, ch0_p1, ..., ch0_pN, ch1_p0, ...]
        # So channel_id = token_id // num_patches_per_channel
        channel_ids = ids_keep // self.num_patches_per_channel

        return latent, ids_keep, channel_ids

    def forward_encoder_dense(
        self, x: torch.Tensor, inherited_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all tokens without dropout removal or artificial masking.

        Runs the full token sequence through the encoder using only the
        inherited (NaN-derived) mask for attention masking.  This produces a
        dense (B, num_patches, embed_dim) output where every token position is
        present.

        NaN values in the input (from missing data) are replaced with zero
        before patch embedding to prevent NaN propagation through the encoder.
        The inherited_mask + attention masking ensure these positions are not
        attended to.

        Args:
            x: Input tensor of shape (B, C, L) e.g. (B, 19, 1440).
            inherited_mask: Optional missing-data mask of shape (B, N).
                If None, assumes no missing data (all zeros).

        Returns:
            latent: Encoded representations of shape (B, num_patches, embed_dim).
            inherited_mask: The mask that was applied, shape (B, N).
        """
        B, _, L = x.shape

        if inherited_mask is None:
            inherited_mask = torch.zeros((B, self.num_patches), device=x.device)

        # Replace NaN with 0 to prevent NaN propagation through patch embedding.
        # The inherited_mask already records which positions had missing data;
        # the attention mask will block these positions from being attended to.
        x = torch.nan_to_num(x, nan=0.0)

        # Patchify and add positional embeddings
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device, dtype=x.dtype)

        N = x.shape[1]

        # Attention mask: block attention TO inherited-masked positions
        attn_mask = torch.zeros(B, 1, 1, N, device=x.device)
        attn_mask.masked_fill_(inherited_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        # Encoder pass on all tokens
        latent = self.encoder(x, attn_mask=attn_mask)

        return latent, inherited_mask

    def forward_decoder(
        self, latent: torch.Tensor, ids_restore: torch.Tensor, total_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            latent: Encoded representations of shape (B, len_keep, embed_dim)
            ids_restore: Indices to restore original order of shape (B, N)
            total_mask: Combined mask of shape (B, N)

        Returns:
            pred: Predictions of shape (B, N, patch_size)
        """
        # Embed
        x = self.encoder_to_decoder(latent)
        B, L_keep, D = x.shape
        L_full = ids_restore.shape[1]

        # Append mask tokens for the *dropped* tokens
        mask_tokens_drop = self.mask_token.repeat(B, L_full - L_keep, 1)
        x_ = torch.cat([x, mask_tokens_drop], dim=1)

        # Unshuffle to original order
        x_full = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # CRITICAL FIX for AIM:
        # The 'latent' contained some tokens that were masked in encoder (via attn_mask).
        # These tokens are garbage. We must replace ALL missing tokens (total_mask=1)
        # with the learnable mask token before decoding.

        # Expand total_mask for replacement
        mask_bool = total_mask.unsqueeze(-1).bool()  # (B, N, 1)

        # Create full sequence of mask tokens
        full_mask_tokens = self.mask_token.repeat(B, L_full, 1)

        # Where mask is True, use mask_token. Where False, use reconstructed sequence.
        x_full = torch.where(mask_bool, full_mask_tokens, x_full)

        # Add pos embed
        x_full = x_full + self.decoder_pos_embed.to(x.device, dtype=x.dtype)

        # Decoder Pass
        x = self.decoder(x_full)

        # Predict
        pred = self.decoder_pred(x)
        return pred

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reconstruction loss on masked patches.

        Args:
            imgs: Original images of shape (B, C, L)
            pred: Predictions of shape (B, N, patch_size)
            mask: Mask indicating which patches to compute loss on (B, N)
            return_per_sample: If True, also return per-sample continuous/binary losses.

        Returns:
            loss: Scalar loss value (when return_per_sample=False)
            (loss, per_sample_dict): Scalar loss + dict with "continuous_loss" and
                "binary_loss" tensors of shape (B,) (when return_per_sample=True)
        """
        target = self.patchify(imgs)
        B, N, P = target.shape

        # Create per-minute mask for zero heart rate values (channel 5)
        # HR patches are at indices [5 * num_patches_per_channel, 6 * num_patches_per_channel)
        hr_start = 5 * self.num_patches_per_channel
        hr_end = 6 * self.num_patches_per_channel
        hr_minute_mask = torch.ones(B, N, P, device=target.device, dtype=target.dtype)
        hr_minute_mask[:, hr_start:hr_end, :] = (target[:, hr_start:hr_end, :] != 0).float()

        if not self.use_hybrid_loss:
            # Original MSE-only loss
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            loss = (pred - target) ** 2
            # Weighted mean to exclude zero HR minutes
            valid_counts = hr_minute_mask.sum(dim=-1).clamp(min=1)
            loss = (loss * hr_minute_mask).sum(dim=-1) / valid_counts
        else:
            # Hybrid loss: MSE for continuous, BCE for binary channels
            # Use float32 for loss computation (numerical stability with AMP)
            pred_f32 = pred.float()
            target_f32 = target.float()
            continuous_mask = self.continuous_patch_mask  # (N,)

            # Normalize only continuous channels if norm_pix_loss is enabled
            if self.norm_pix_loss:
                # Only normalize continuous channel patches
                cont_target = target_f32[:, continuous_mask, :]  # (B, N_cont, P)
                mean = cont_target.mean(dim=-1, keepdim=True)
                var = cont_target.var(dim=-1, keepdim=True)
                cont_target_norm = (cont_target - mean) / (var + 1.0e-6) ** 0.5
                target_f32 = target_f32.clone()
                target_f32[:, continuous_mask, :] = cont_target_norm

            # MSE loss for all patches (will only use continuous)
            # Weighted mean to exclude zero HR minutes
            mse_loss = (pred_f32 - target_f32) ** 2
            hr_minute_mask_f32 = hr_minute_mask.float()
            valid_counts = hr_minute_mask_f32.sum(dim=-1).clamp(min=1)
            mse_per_patch = (mse_loss * hr_minute_mask_f32).sum(dim=-1) / valid_counts  # (B, N)

            # BCE loss for binary channel patches
            binary_mask = ~continuous_mask  # (N,)
            pred_binary = pred_f32[:, binary_mask, :]  # (B, N_binary, P)
            target_binary = target_f32[:, binary_mask, :]  # (B, N_binary, P)

            # Clamp targets to [0, 1] for BCE (should already be binary, but safety)
            target_binary = target_binary.clamp(0.0, 1.0)

            bce_per_patch = F.binary_cross_entropy_with_logits(
                pred_binary, target_binary, reduction="none"
            ).mean(dim=-1)  # (B, N_binary)

            # Combine: MSE for continuous, BCE for binary
            loss = torch.zeros(B, N, device=pred.device, dtype=torch.float32)
            loss[:, continuous_mask] = mse_per_patch[:, continuous_mask]
            loss[:, binary_mask] = bce_per_patch

            # Apply per-channel weights
            loss = loss * self.patch_weights  # (B, N) * (N,) -> (B, N)

        # Optionally compute per-sample split losses before scalar reduction
        per_sample_losses = None
        if return_per_sample:
            cont_mask_vals = mask * self.continuous_patch_mask.float()  # (B, N)
            per_sample_cont = (loss * cont_mask_vals).sum(dim=1) / cont_mask_vals.sum(dim=1).clamp(
                min=1
            )

            bin_mask_vals = mask * (~self.continuous_patch_mask).float()  # (B, N)
            per_sample_bin = (loss * bin_mask_vals).sum(dim=1) / bin_mask_vals.sum(dim=1).clamp(
                min=1
            )

            per_sample_losses = {
                "continuous_loss": per_sample_cont,
                "binary_loss": per_sample_bin,
            }

        # Loss only on mask=1
        if mask.sum() == 0:
            scalar_loss = loss.mean()
        else:
            scalar_loss = (loss * mask).sum() / mask.sum()

        if return_per_sample:
            return scalar_loss, per_sample_losses
        return scalar_loss

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        Args:
            imgs: Input tensor of shape (B, C, L) e.g. (B, 19, 1440)

        Returns:
            Patches of shape (B, num_patches, patch_size) e.g. (B, 2736, 10)
        """
        p = self.patch_size
        B, C, L = imgs.shape
        x = imgs.reshape(B, C, L // p, p)
        x = x.permute(0, 1, 2, 3).contiguous().view(B, -1, p)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse patchify operation.

        Args:
            x: Patches of shape (B, num_patches, patch_size) e.g. (B, 2736, 10)

        Returns:
            Tensor of shape (B, C, L) e.g. (B, 19, 1440)
        """
        p = self.patch_size
        h = self.in_channels
        w = self.num_patches_per_channel
        B = x.shape[0]

        # Reshape: (B, 2736, 10) -> (B, 19, 144, 10)
        x = x.view(B, h, w, p)
        # Merge patches: (B, 19, 144, 10) -> (B, 19, 1440)
        x = x.contiguous().view(B, h, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        inherited_mask: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Forward pass through MAE.

        Args:
            x: Input tensor.
            inherited_mask: Optional inherited mask.
            return_per_sample: Whether to return per-sample losses.

        Returns:
            Tuple of loss tensors and optionally per-sample losses.
        """
        if inherited_mask is None:
            B, _, L = x.shape
            N = self.num_patches
            inherited_mask = torch.zeros((B, N), device=x.device)

        latent, total_mask, artificial_mask, ids_restore = self.forward_encoder(x, inherited_mask)
        pred = self.forward_decoder(latent, ids_restore, total_mask)

        # Loss only where we artificially masked REAL data
        loss_mask = artificial_mask * (1 - inherited_mask)
        loss_result = self.forward_loss(x, pred, loss_mask, return_per_sample=return_per_sample)

        if return_per_sample:
            loss, per_sample_losses = loss_result
            return loss, pred, total_mask, per_sample_losses
        return loss_result, pred, total_mask
