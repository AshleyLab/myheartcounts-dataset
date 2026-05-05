"""Channel-wise StandardScaler utilities for forecasting training caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ChannelStandardScalerStats:
    """Train-fit per-channel StandardScaler statistics."""

    means: np.ndarray
    stds: np.ndarray
    valid_counts: np.ndarray
    fit_scope: str = "train"

    def __post_init__(self) -> None:
        """Normalize scaler statistics to stable NumPy dtypes."""
        object.__setattr__(self, "means", np.asarray(self.means, dtype=np.float64))
        object.__setattr__(self, "stds", np.asarray(self.stds, dtype=np.float64))
        object.__setattr__(self, "valid_counts", np.asarray(self.valid_counts, dtype=np.int64))

    @property
    def n_channels(self) -> int:
        """Return the number of channel statistics stored."""
        return int(self.means.shape[0])

    def transform_history_cf(self, history_cf: torch.Tensor) -> torch.Tensor:
        """Standardize a channel-first history tensor."""
        return transform_history_cf(history_cf, self)

    def inverse_transform_history_cf(self, history_cf: torch.Tensor) -> torch.Tensor:
        """Restore a standardized channel-first history tensor."""
        return inverse_transform_history_cf(history_cf, self)

    def save_stats_json(self, output_path: str | Path) -> Path:
        """Persist these scaler statistics to JSON."""
        return save_stats_json(self, output_path)


def fit_from_history_cf_rows(
    history_cf_rows: list[torch.Tensor],
    *,
    n_channels: int | None = None,
) -> ChannelStandardScalerStats:
    """Fit per-channel mean/std from train-split history_cf rows only."""
    if not history_cf_rows:
        if n_channels is None:
            raise ValueError("n_channels is required when fitting scaler on an empty row list.")
        means = np.zeros(n_channels, dtype=np.float64)
        stds = np.ones(n_channels, dtype=np.float64)
        valid_counts = np.zeros(n_channels, dtype=np.int64)
        return ChannelStandardScalerStats(means=means, stds=stds, valid_counts=valid_counts)

    if n_channels is None:
        n_channels = int(history_cf_rows[0].shape[0])

    value_sum = np.zeros(n_channels, dtype=np.float64)
    value_sq_sum = np.zeros(n_channels, dtype=np.float64)
    valid_counts = np.zeros(n_channels, dtype=np.int64)

    for row in history_cf_rows:
        row_np = row.detach().cpu().numpy().astype(np.float64, copy=False)
        valid_mask = ~np.isnan(row_np)
        valid_counts += valid_mask.sum(axis=1).astype(np.int64)
        value_sum += np.where(valid_mask, row_np, 0.0).sum(axis=1)
        value_sq_sum += np.where(valid_mask, row_np * row_np, 0.0).sum(axis=1)

    means = np.zeros(n_channels, dtype=np.float64)
    non_empty_mask = valid_counts > 0
    means[non_empty_mask] = value_sum[non_empty_mask] / valid_counts[non_empty_mask]

    variances = np.zeros(n_channels, dtype=np.float64)
    variances[non_empty_mask] = (
        value_sq_sum[non_empty_mask] / valid_counts[non_empty_mask] - means[non_empty_mask] ** 2
    )
    variances = np.maximum(variances, 0.0)
    stds = np.ones(n_channels, dtype=np.float64)
    stds[non_empty_mask] = np.sqrt(variances[non_empty_mask])
    stds[stds == 0.0] = 1.0

    return ChannelStandardScalerStats(
        means=means,
        stds=stds,
        valid_counts=valid_counts,
    )


def transform_history_cf(
    history_cf: torch.Tensor,
    stats: ChannelStandardScalerStats,
) -> torch.Tensor:
    """Apply channel-wise StandardScaler to a channel-first tensor, preserving NaN."""
    output = history_cf.clone()
    means = torch.as_tensor(stats.means, dtype=output.dtype, device=output.device).unsqueeze(1)
    stds = torch.as_tensor(stats.stds, dtype=output.dtype, device=output.device).unsqueeze(1)
    valid_mask = ~torch.isnan(output)
    output[valid_mask] = ((output - means) / stds)[valid_mask]
    return output


def inverse_transform_history_cf(
    history_cf: torch.Tensor,
    stats: ChannelStandardScalerStats,
) -> torch.Tensor:
    """Invert channel-wise StandardScaler on a channel-first tensor, preserving NaN."""
    output = history_cf.clone()
    means = torch.as_tensor(stats.means, dtype=output.dtype, device=output.device).unsqueeze(1)
    stds = torch.as_tensor(stats.stds, dtype=output.dtype, device=output.device).unsqueeze(1)
    valid_mask = ~torch.isnan(output)
    output[valid_mask] = (output * stds + means)[valid_mask]
    return output


def save_stats_json(
    stats: ChannelStandardScalerStats,
    output_path: str | Path,
) -> Path:
    """Persist scaler statistics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fit_scope": stats.fit_scope,
        "n_channels": stats.n_channels,
        "means": stats.means.tolist(),
        "stds": stats.stds.tolist(),
        "valid_counts": stats.valid_counts.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_stats_json(input_path: str | Path) -> ChannelStandardScalerStats:
    """Load scaler statistics from JSON."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return ChannelStandardScalerStats(
        means=np.asarray(payload["means"], dtype=np.float64),
        stds=np.asarray(payload["stds"], dtype=np.float64),
        valid_counts=np.asarray(payload["valid_counts"], dtype=np.int64),
        fit_scope=str(payload.get("fit_scope", "train")),
    )
