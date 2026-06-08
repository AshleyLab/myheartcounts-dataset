"""Bi-directional Mamba2 week encoder (WBM) — inference architecture.

Only the encoder + forward pass are needed downstream: load the pretrained
checkpoint and run a forward pass on weekly tensors to get the 256-d
representation ``r``. Training (contrastive loss, Lightning module) is not
included here. ``mamba_ssm`` is a CUDA-only optional dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HourPatchEmbedding

try:
    # adjust import based on the installed package
    from mamba_ssm import Mamba2  # common pattern
except Exception:
    Mamba2 = None


class BiMamba2Block(nn.Module):
    """Bidirectional Mamba2 block with FFN and residual connections."""

    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.0):
        """Initialize BiMamba2Block."""
        super().__init__()
        if Mamba2 is None:
            raise ImportError("mamba-ssm is not installed or Mamba2 import failed.")

        self.fwd = Mamba2(d_model=d_model)
        self.bwd = Mamba2(d_model=d_model)
        self.proj = nn.Linear(2 * d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: bidirectional Mamba2 + FFN."""
        # x: (B, T, D)
        y_f = self.fwd(x)  # (B, T, D)
        y_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])  # (B, T, D)

        y = self.proj(torch.cat([y_f, y_b], dim=-1))
        x = self.norm1(x + self.drop1(y))

        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class Mamba2WeekEncoder(nn.Module):
    """Tokenizer + Bi-Mamba2 backbone + projection head.

    Input:  (B, 168, in_dim)  — default (B, 168, 38) without time features
    Output:
      - h: (B, proj_dim) normalized projection vector (for contrastive loss)
      - r: (B, embed_dim) pooled representation before projection (for downstream tasks)
    """

    def __init__(  # noqa: D107
        self,
        in_dim: int = 38,  # 19 values + 19 masks (no time features)
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        proj_dim: int = 128,
        dropout: float = 0.05,
        ffn_mult: int = 4,
        proj_head_type: str = "mlp",  # "linear" or "mlp"
    ):
        super().__init__()

        self.tokenizer = HourPatchEmbedding(
            in_dim=in_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

        self.backbone = nn.Sequential(
            *[
                BiMamba2Block(embed_dim, ffn_mult=ffn_mult, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        if proj_head_type == "linear":
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
            )
        else:  # "mlp" — 3-layer projector
            proj_hidden = 4 * embed_dim
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, proj_hidden),  # D -> 4D
                nn.LayerNorm(proj_hidden),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(proj_hidden, embed_dim),  # 4D -> D
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(embed_dim, proj_dim),  # D -> proj
            )

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor | None = None):
        """Forward pass through the Mamba2 week encoder.

        Args:
            x: Input tensor of shape (B, 168, in_dim).
            keep_mask: Optional binary mask (B, 168) where 1=keep, 0=drop.
                       If None, all tokens are kept.

        Returns:
            Tuple of (h, r) where:
            h: (B, proj_dim) normalized projection (for contrastive loss)
            r: (B, embed_dim) pooled representation before projection (for downstream tasks)
        """
        # x: (B, 168, in_dim)
        tok = self.tokenizer(x)  # (B, 168, embed_dim)
        seq = self.backbone(tok)  # (B, 168, embed_dim)

        # Masked pooling over time
        if keep_mask is not None:
            # Masked mean: exclude dropped tokens from pooling
            mask_expanded = keep_mask.unsqueeze(-1)  # (B, 168, 1)
            seq_masked = seq * mask_expanded
            r = seq_masked.sum(dim=1) / keep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            # Simple mean pooling (no masking)
            r = seq.mean(dim=1)  # (B, embed_dim)

        h_raw = self.proj_head(r)  # (B, proj_dim)
        h = F.normalize(h_raw, dim=-1)  # cosine space for InfoNCE
        return h, r
