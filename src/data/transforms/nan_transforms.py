"""Tensor transforms for handling NaN values in daily data preprocessing."""

import math

import torch

__all__ = [
    "FillNaN",
    "ZeroToNaNTransform",
    "HybridNaNAwareNormalize",
]


class FillNaN:
    """Fill NaN values with a specified value.

    Args:
        fill_value: Value to replace NaN with. Defaults to 0.0.

    Example:
        >>> fill = FillNaN(0.0)
        >>> x = torch.tensor([1.0, float('nan'), 3.0])
        >>> filled = fill(x)  # tensor([1., 0., 3.])
    """

    def __init__(self, fill_value: float = 0.0):
        """Initialize the FillNaN transform.

        Args:
            fill_value: Value to replace NaN with.
        """
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN values in tensor.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with NaN values replaced by fill_value.
        """
        return torch.where(torch.isnan(x), self.fill_value, x)


class HybridNaNAwareNormalize:
    """Normalize using a Bayesian hybrid of instance and global statistics.

    Computes a weighted average of the instance mean/std and global prior mean/std
    based on the number of valid (non-NaN) elements.

    mean_est = (N * mean_instance + lambda * mean_prior) / (N + lambda)

    - When N is large (lots of data), approaches instance normalization.
    - When N is small (sparse data), approaches global normalization.
    - When prior_count is 0, performs pure instance normalization.
    - When prior_count is inf, performs pure global normalization.

    Args:
        channels: List of channel indices to normalize.
        mean_prior: Global mean prior (C,). Optional (required if prior_count > 0).
        std_prior: Global std prior (C,). Optional (required if prior_count > 0).
        prior_count: Strength of the prior (lambda). Defaults to 0.0 (pure instance norm).
        epsilon: Small constant for stability.

    Example:
        >>> # Pure Instance Normalization (default)
        >>> norm = HybridNaNAwareNormalize(channels=[0])
        >>>
        >>> # Hybrid Normalization (prior strength = 4 hours)
        >>> norm = HybridNaNAwareNormalize(
        ...     channels=[0],
        ...     mean_prior=[50.0],
        ...     std_prior=[10.0],
        ...     prior_count=240.0
        ... )
        >>>
        >>> # Pure Global Normalization
        >>> norm = HybridNaNAwareNormalize(
        ...     channels=[0],
        ...     mean_prior=[50.0],
        ...     std_prior=[10.0],
        ...     prior_count=float('inf')
        ... )
    """

    def __init__(
        self,
        channels: list[int],
        mean_prior: torch.Tensor | list[float] | None = None,
        std_prior: torch.Tensor | list[float] | None = None,
        prior_count: float = 0.0,
        epsilon: float = 1e-8,
    ):
        """Initialize the hybrid normalization transform.

        Args:
            channels: Channel indices to normalize.
            mean_prior: Global mean prior.
            std_prior: Global std prior.
            prior_count: Strength of the prior.
            epsilon: Stability constant.
        """
        self.channels = channels
        self.prior_count = float(prior_count)
        self.epsilon = epsilon

        self.mean_prior: torch.Tensor | None
        self.std_prior: torch.Tensor | None
        if self.prior_count > 0:
            if mean_prior is None or std_prior is None:
                raise ValueError("mean_prior and std_prior must be provided if prior_count > 0")
            self.mean_prior = torch.as_tensor(mean_prior).float()
            self.std_prior = torch.as_tensor(std_prior).float()
        else:
            self.mean_prior = None
            self.std_prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using hybrid statistics.

        Args:
            x: Input tensor of shape (C, T) or (B, C, T).

        Returns:
            Normalized tensor.
        """
        result = x.clone()
        channels_to_norm = result[self.channels]

        # 1. Handle Pure Global Normalization (Infinite or Large Prior)
        if math.isinf(self.prior_count) or self.prior_count >= 1e12:
            # Ensure priors are on correct device and sliced to match channels
            mean_prior = self.mean_prior[self.channels].to(x.device).view(-1, 1)
            std_prior = self.std_prior[self.channels].to(x.device).view(-1, 1)

            normalized = (channels_to_norm - mean_prior) / (std_prior + self.epsilon)
            result[self.channels] = normalized
            return result

        # 2. Compute Instance Statistics
        mask = ~torch.isnan(channels_to_norm)
        x_masked = torch.where(mask, channels_to_norm, torch.zeros_like(channels_to_norm))
        count = mask.sum(dim=-1, keepdim=True).float()  # (C, 1)

        # Avoid div by zero in instance mean
        instance_mean = x_masked.sum(dim=-1, keepdim=True) / count.clamp(min=1)

        # Instance Variance
        centered = torch.where(
            mask, (channels_to_norm - instance_mean).pow(2), torch.zeros_like(channels_to_norm)
        )
        instance_var = centered.sum(dim=-1, keepdim=True) / (count - 1).clamp(min=1)

        # 2. Compute Hybrid Statistics (MAP Estimate)
        if self.prior_count > 0 and self.mean_prior is not None:
            # Ensure priors are on correct device and sliced to match channels
            mean_prior = self.mean_prior[self.channels].to(x.device).view(-1, 1)
            std_prior = self.std_prior[self.channels].to(x.device).view(-1, 1)

            hybrid_mean = (count * instance_mean + self.prior_count * mean_prior) / (
                count + self.prior_count
            )
            prior_var = std_prior.pow(2)
            hybrid_var = (count * instance_var + self.prior_count * prior_var) / (
                count + self.prior_count
            )
            hybrid_std = hybrid_var.sqrt()
        else:
            # Pure instance normalization
            hybrid_mean = instance_mean
            hybrid_std = instance_var.sqrt()

        # 3. Apply Normalization
        normalized = (channels_to_norm - hybrid_mean) / (hybrid_std + self.epsilon)

        result[self.channels] = normalized
        return result

    def denormalize(self, z: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse normalization (inverse of __call__).

        When prior_count >= 1e12 (global norm), uses stored priors.
        Otherwise requires x to recompute the same stats used during normalize.

        Args:
            z: Normalized tensor of shape (C, T) or (B, C, T).
            x: Original input used for normalize (required when prior_count < 1e12).

        Returns:
            Denormalized tensor of same shape as z.
        """
        result = z.clone()
        channels_to_denorm = result[self.channels]

        if math.isinf(self.prior_count) or self.prior_count >= 1e12:
            assert self.mean_prior is not None and self.std_prior is not None
            mean_prior = self.mean_prior[self.channels].to(z.device).view(-1, 1)
            std_prior = self.std_prior[self.channels].to(z.device).view(-1, 1)
            result[self.channels] = channels_to_denorm * (std_prior + self.epsilon) + mean_prior
            return result

        if x is None:
            raise ValueError("x required for denormalize when prior_count < 1e12 (hybrid/instance)")

        # Recompute same stats as __call__
        channels_from_x = x[self.channels]
        mask = ~torch.isnan(channels_from_x)
        x_masked = torch.where(mask, channels_from_x, torch.zeros_like(channels_from_x))
        count = mask.sum(dim=-1, keepdim=True).float()
        instance_mean = x_masked.sum(dim=-1, keepdim=True) / count.clamp(min=1)
        centered = torch.where(
            mask, (channels_from_x - instance_mean).pow(2), torch.zeros_like(channels_from_x)
        )
        instance_var = centered.sum(dim=-1, keepdim=True) / (count - 1).clamp(min=1)
        instance_std = instance_var.sqrt()

        if self.prior_count > 0 and self.mean_prior is not None:
            mean_prior = self.mean_prior[self.channels].to(x.device).view(-1, 1)
            std_prior = self.std_prior[self.channels].to(x.device).view(-1, 1)
            hybrid_mean = (count * instance_mean + self.prior_count * mean_prior) / (
                count + self.prior_count
            )
            prior_var = std_prior.pow(2)
            hybrid_var = (count * instance_var + self.prior_count * prior_var) / (
                count + self.prior_count
            )
            hybrid_std = hybrid_var.sqrt()
        else:
            hybrid_mean = instance_mean
            hybrid_std = instance_std

        result[self.channels] = channels_to_denorm * (hybrid_std + self.epsilon) + hybrid_mean
        return result


class ZeroToNaNTransform:
    """Pre-normalization transform to handle zero values as missing data.

    Converts specific zero values to NaN before normalization:
    - Heart rate (ch 5): Replace per-value 0 with NaN (0 bpm is physiologically invalid)
    - Steps/distance/energy (ch 0, 1, 3, 4, 6): Replace entire channel with NaN if all
      zeros (device not worn/carried). Individual zeros are valid (e.g. sitting still).
    - Flights climbed (ch 2): Untouched — ~55% of samples are legitimately all-zero.

    Args:
        heart_rate_idx: Channel index for heart rate. Default: 5.
        all_zero_nan_channels: Channel indices where an all-zero channel is converted
            entirely to NaN (device not worn). Default: (0, 1, 3, 4, 6).
        short_sleep_nan_channels: Channel indices for sleep channels where
            short or absent sessions indicate unreliable supervision. When the
            total sleep for a channel is below ``short_sleep_threshold``
            (including zero — no sleep logged at all, likely watch off the
            wrist overnight), the zero portions are set to NaN. Default:
            (7, 8).
        short_sleep_threshold: Minimum total sleep (in minutes) for a day's sleep
            data to be considered valid. At or below this, zeros are replaced
            with NaN. Default: 180.0 (3 hours).

    Example:
        >>> transform = ZeroToNaNTransform()
        >>> x = torch.zeros(19, 1440)
        >>> x[5, :10] = 60.0  # Some valid heart rate values
        >>> transformed = transform(x)
        >>> torch.isnan(transformed[5, 10:]).all()  # Zero HR values are now NaN
        True
        >>> torch.isnan(transformed[6]).all()  # All-zero active energy is all NaN
        True
        >>> torch.isnan(transformed[0]).all()  # All-zero steps also all NaN
        True
    """

    def __init__(
        self,
        heart_rate_idx: int = 5,
        all_zero_nan_channels: tuple[int, ...] = (0, 1, 3, 4, 6),
        short_sleep_nan_channels: tuple[int, ...] = (7, 8),
        short_sleep_threshold: float = 180.0,
    ):
        """Initialize the zero to NaN transform.

        Args:
            heart_rate_idx: Channel index for heart rate.
            all_zero_nan_channels: Channel indices where all-zero → all-NaN.
            short_sleep_nan_channels: Sleep channel indices for short-session handling.
            short_sleep_threshold: Minimum total sleep minutes for valid detection.
        """
        self.heart_rate_idx = heart_rate_idx
        self.all_zero_nan_channels = all_zero_nan_channels
        self.short_sleep_nan_channels = short_sleep_nan_channels
        self.short_sleep_threshold = short_sleep_threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Convert zero values to NaN for heart rate and all-zero activity channels.

        Args:
            x: Input tensor of shape (C, T).

        Returns:
            Tensor with zeros converted to NaN in specified channels.
        """
        x = x.clone()

        # Heart rate: set all 0 values to NaN
        hr = x[self.heart_rate_idx]
        x[self.heart_rate_idx] = torch.where(hr == 0, float("nan"), hr)

        # Steps/distance/energy: set entire channel to NaN if all zeros (or NaN)
        for ch_idx in self.all_zero_nan_channels:
            ch = x[ch_idx]
            if ((ch == 0) | torch.isnan(ch)).all():
                x[ch_idx] = float("nan")

        # Sleep channels: set zeros to NaN if total sleep < threshold. This
        # covers both short detected sessions (likely detection error) and
        # zero-sleep days (watch likely off-wrist overnight) — in neither case
        # is the "awake" signal a trustworthy ground-truth target.
        for ch_idx in self.short_sleep_nan_channels:
            ch = x[ch_idx]
            total_sleep = ch[~torch.isnan(ch)].sum().item()
            if total_sleep < self.short_sleep_threshold:
                x[ch_idx] = torch.where(ch == 0, float("nan"), ch)

        return x
