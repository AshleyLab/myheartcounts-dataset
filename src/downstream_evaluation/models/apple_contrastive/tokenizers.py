"""Tokenizers for converting time series data into embeddings."""

import torch.nn as nn
import torch.nn.functional as F


class HourPatchEmbedding(nn.Module):
    """TST-style patch embedding used in WBM.

    LayerNorm -> Linear -> GELU -> Linear
    """

    def __init__(self, in_dim=38, embed_dim=256, hidden_dim=256, dropout=0.0):
        """Initialize hour patch embedding.

        Args:
            in_dim: Input dimension.
            embed_dim: Embedding dimension.
            hidden_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the embedding.

        Args:
            x: Input tensor.

        Returns:
            Embedded tensor.
        """
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return h
