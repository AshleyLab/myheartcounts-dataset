"""Transformer building blocks for MAE-ViT 1D."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding for 19 channels x 1440 minutes."""

    def __init__(
        self,
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
    ):
        """Initialize 1D patch embedding.

        Args:
            seq_length: Sequence length.
            patch_size: Patch size.
            in_channels: Input channels.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches_per_channel = seq_length // patch_size
        self.total_patches = self.num_patches_per_channel * in_channels

        # Shared projection layer
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, L)

        Returns:
            Patch embeddings of shape (B, Total_Patches, embed_dim)
        """
        B, C, L = x.shape
        assert L == self.seq_length, f"Input length {L} != {self.seq_length}"
        assert C == self.in_channels, f"Input channels {C} != {self.in_channels}"

        # Flatten channels into batch dimension for shared projection
        x = x.view(B * C, 1, L)
        x = self.proj(x)  # (B*C, embed_dim, num_patches_per_channel)

        # Reshape to (B, Total_Patches, embed_dim)
        x = x.view(B, C, self.embed_dim, self.num_patches_per_channel)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, P_per_ch, embed_dim)
        x = x.view(B, self.total_patches, self.embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional attention masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of heads.
            dropout: Dropout rate.
            qkv_bias: Whether to use bias term in the QKV projection.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope_fn=None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            attn_mask: Optional attention mask. Can be:
                - (B, N) where True/1 is "keep" and False/0 is "mask"
                - (B, 1, 1, N) additive mask with -inf for masked positions
            rope_fn: Optional callable (q, k) -> (q_rot, k_rot) for rotary pos emb.

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_fn is not None:
            q, k = rope_fn(q, k)

        # Use PyTorch SDPA (Flash/Memory-Efficient/Math) to avoid materializing (B, H, N, N).
        sdpa_mask = None
        if attn_mask is not None:
            # Supported inputs in this codebase:
            # - additive mask (B, 1, 1, N) with 0 for valid keys and -inf for masked keys
            # - additive mask (B, N) (treated as key mask, broadcast across queries/heads)
            if attn_mask.dim() == 2:
                sdpa_mask = attn_mask[:, None, None, :]  # (B, 1, 1, N)
            else:
                sdpa_mask = attn_mask

            # Broadcast key mask across query length to match SDPA's expected shape.
            # (B, 1, 1, K) -> (B, 1, Q, K)
            if sdpa_mask.shape[-2] == 1:
                sdpa_mask = sdpa_mask.expand(B, 1, N, N)

            sdpa_mask = sdpa_mask.to(dtype=q.dtype, device=q.device)

        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP ratio.
            dropout: Dropout rate.
            qkv_bias: Whether to use bias term in the QKV projection.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope_fn=None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            attn_mask: Optional attention mask
            rope_fn: Optional callable (q, k) -> (q_rot, k_rot) for rotary pos emb.

        Returns:
            Output tensor of shape (B, N, C)
        """
        x = x + self.attn(self.norm1(x), attn_mask, rope_fn=rope_fn)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with attention masking support."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        """Initialize ViT encoder.

        Args:
            embed_dim: Embedding dimension.
            depth: Encoder depth.
            num_heads: Number of attention heads.
            mlp_ratio: MLP ratio.
            dropout: Dropout rate.
            qkv_bias: Whether to use bias term in the QKV projection.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            attn_mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        return x
