"""Masking module for imputation evaluation.

Provides various mask generators for simulating missing data scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imputation_evaluation.config import MaskingConfig
    from imputation_evaluation.masking.base import MaskGenerator
    from imputation_evaluation.masking.generator import (
        MaskCache,
        MaskCacheGenerator,
        ScenarioMasks,
    )

__all__ = [
    "MaskGenerator",
    "MaskResult",
    "create_mask_generators",
    "RandomNoiseMask",
    "TemporalSliceMask",
    "SignalSliceMask",
    "SleepGapMask",
    "WorkoutGapMask",
    "IntensityFailureMask",
    # Mask cache generation
    "MaskCache",
    "MaskCacheGenerator",
    "ScenarioMasks",
]


def __getattr__(name: str):
    """Lazy import."""
    if name in ("MaskGenerator", "MaskResult"):
        from imputation_evaluation.masking.base import MaskGenerator, MaskResult

        return MaskGenerator if name == "MaskGenerator" else MaskResult
    elif name in ("MaskCache", "MaskCacheGenerator", "ScenarioMasks"):
        from imputation_evaluation.masking.generator import (
            MaskCache,
            MaskCacheGenerator,
            ScenarioMasks,
        )

        if name == "MaskCache":
            return MaskCache
        elif name == "MaskCacheGenerator":
            return MaskCacheGenerator
        else:
            return ScenarioMasks
    elif name == "RandomNoiseMask":
        from imputation_evaluation.masking.random_noise import RandomNoiseMask

        return RandomNoiseMask
    elif name == "TemporalSliceMask":
        from imputation_evaluation.masking.temporal_slice import TemporalSliceMask

        return TemporalSliceMask
    elif name == "SignalSliceMask":
        from imputation_evaluation.masking.signal_slice import SignalSliceMask

        return SignalSliceMask
    elif name == "SleepGapMask":
        from imputation_evaluation.masking.sleep_gap import SleepGapMask

        return SleepGapMask
    elif name == "WorkoutGapMask":
        from imputation_evaluation.masking.workout_gap import WorkoutGapMask

        return WorkoutGapMask
    elif name == "IntensityFailureMask":
        from imputation_evaluation.masking.intensity_failure import IntensityFailureMask

        return IntensityFailureMask
    elif name == "create_mask_generators":
        # Inline to avoid circular import
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_mask_generators(config: MaskingConfig) -> list[MaskGenerator]:
    """Create mask generators from configuration.

    Args:
        config: Masking configuration.

    Returns:
        List of enabled mask generators.
    """
    from imputation_evaluation.masking.intensity_failure import IntensityFailureMask
    from imputation_evaluation.masking.random_noise import RandomNoiseMask
    from imputation_evaluation.masking.signal_slice import SignalSliceMask
    from imputation_evaluation.masking.sleep_gap import SleepGapMask
    from imputation_evaluation.masking.temporal_slice import TemporalSliceMask
    from imputation_evaluation.masking.workout_gap import WorkoutGapMask

    generators: list[MaskGenerator] = []

    # Tier 1: Structural masks
    if config.random_noise.enabled:
        generators.append(
            RandomNoiseMask(
                patch_size=config.random_noise.patch_size,
                mask_ratio=config.random_noise.mask_ratio,
            )
        )

    if config.temporal_slice.enabled:
        generators.append(
            TemporalSliceMask(
                mask_ratio=config.temporal_slice.mask_ratio,
                min_block_size=config.temporal_slice.min_block_size,
                max_block_size=config.temporal_slice.max_block_size,
            )
        )

    if config.signal_slice.enabled:
        generators.append(
            SignalSliceMask(
                mask_ratio=config.signal_slice.mask_ratio,
                device_groups=config.signal_slice.device_groups,
            )
        )

    # Tier 2: Semantic masks
    if config.sleep_gap.enabled:
        generators.append(
            SleepGapMask(
                asleep_channel=config.sleep_gap.asleep_channel,
                inbed_channel=config.sleep_gap.inbed_channel,
            )
        )

    if config.workout_gap.enabled:
        generators.append(
            WorkoutGapMask(
                mask_channels=config.workout_gap.mask_channels,
                workout_channels=config.workout_gap.workout_channels,
            )
        )

    if config.intensity_failure.enabled:
        generators.append(
            IntensityFailureMask(
                hr_channel=config.intensity_failure.hr_channel,
                hr_threshold=config.intensity_failure.hr_threshold,
                hr_unit=config.intensity_failure.hr_unit,
                mask_channels=config.intensity_failure.mask_channels,
            )
        )

    return generators
