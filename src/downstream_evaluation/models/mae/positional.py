"""Positional embedding utilities for MAE-ViT 1D."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal position embeddings.

    Args:
        embed_dim: output dimension for each position
        pos: list of positions to be encoded: size (M,)

    Returns:
        out: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim/2), outer product

    emb_sin = np.sin(out)  # (M, embed_dim/2)
    emb_cos = np.cos(out)  # (M, embed_dim/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RotaryDayEmbedding(nn.Module):
    """Rotary positional embedding for day positions in cross-day attention."""

    def __init__(self, head_dim: int, max_positions: int = 365, theta: float = 10000.0):  # noqa: D107
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_table", angles.cos(), persistent=False)
        self.register_buffer("sin_table", angles.sin(), persistent=False)

    def get_rope_fn(  # noqa: D102
        self, positions: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if positions.dim() == 1:
            cos = self.cos_table[positions].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D/2)
            sin = self.sin_table[positions].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_table[positions].unsqueeze(1)  # (B, 1, N, D/2)
            sin = self.sin_table[positions].unsqueeze(1)

        def apply_rope(
            q: torch.Tensor, k: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            c = cos.to(dtype=q.dtype)
            s = sin.to(dtype=q.dtype)
            return _apply_rotary(q, c, s), _apply_rotary(k, c, s)

        return apply_rope
