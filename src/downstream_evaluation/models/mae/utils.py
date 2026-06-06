"""Utility functions for MAE-ViT 1D."""

import torch


def create_inherited_mask(x: torch.Tensor, patch_size: int = 10) -> torch.Tensor:
    """Create inherited mask from NaN values in raw data.

    Detects missing data at the patch level:
    - Default: If ANY value within a patch is NaN, the entire patch is marked as missing
    - Heart rate (channel 5): Only marked as missing if ALL values are NaN
      (handles sporadic zero HR values that become NaN via ZeroToNaNTransform)

    Args:
        x: Raw input data of shape (B, C, L) e.g. (B, 19, 1440)
        patch_size: Size of each patch (default: 10 minutes)

    Returns:
        inherited_mask: Tensor of shape (B, num_patches) where
            1 = Missing (contains NaN), 0 = Present (all valid)
            num_patches = C * (L // patch_size) e.g. 19 * 144 = 2736
    """
    B, C, L = x.shape

    # Reshape to patches: (B, C, num_patches_per_channel, patch_size)
    x_patched = x.view(B, C, L // patch_size, patch_size)

    # Check for missingness in each patch
    # Default: ANY NaN → missing
    is_missing = torch.isnan(x_patched).any(dim=-1)  # (B, C, num_patches_per_channel)

    # Heart rate (channel 5): ALL NaN → missing
    # This handles sporadic zero HR values that become NaN via ZeroToNaNTransform
    hr_all_nan = torch.isnan(x_patched[:, 5, :, :]).all(dim=-1)  # (B, num_patches_per_channel)
    is_missing[:, 5, :] = hr_all_nan

    # Flatten to match the Transformer sequence order: (B, C * num_patches_per_channel)
    # This matches the permutation used in PatchEmbed1D which does:
    # x.view(B, C, ...).permute(0, 1, ...).view(B, Total, ...)
    inherited_mask = is_missing.view(B, -1).float()

    return inherited_mask
