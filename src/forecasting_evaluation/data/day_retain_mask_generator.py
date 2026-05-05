"""Generate and persist day-level retain masks for forecasting sample selection."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import datasets as hf_ds

if TYPE_CHECKING:
    from data.filters.daily_filters import Filter

logger = logging.getLogger(__name__)


def generate_day_drop_mask(
    daily_hf_dir: str | Path,
    min_wear_fraction: float = 0.5,
    variance_filter_enabled: bool = True,
    variance_thresholds: dict[int, float] | None = None,
    num_proc: int = 4,
    use_cache: bool = False,
) -> dict[str, list[str]]:
    """Generate a keep-mask for day-level samples after QA filtering.

    The returned mask contains only days that survive filtering:
    ``{user_id: [date1, date2, ...]}``.

    Args:
        daily_hf_dir: Path to daily HuggingFace dataset directory.
        min_wear_fraction: Minimum wear-time fraction for ``WearTimeFilter``.
        variance_filter_enabled: Whether to apply ``LowChannelVarianceFilter``.
        variance_thresholds: Optional channel variance thresholds.
        num_proc: Number of processes passed to ``apply_filters``.
        use_cache: Whether to load HF filter results from cache.

    Returns:
        Dict mapping user_id to sorted list of kept day timestamps (date strings).
    """
    print(f"Loading daily HF dataset from {daily_hf_dir}")
    logger.info(f"Loading daily HF dataset from {daily_hf_dir}")
    ds = hf_ds.load_from_disk(daily_hf_dir)
    logger.info(f"Loaded {len(ds)} samples")

    # Import lazily to avoid circular import during module initialization.
    from data.filters.daily_filters import LowChannelVarianceFilter, WearTimeFilter, apply_filters

    filters: list[Filter] = []
    if min_wear_fraction > 0.0:
        filters.append(WearTimeFilter(min_wear_fraction=min_wear_fraction))

    if variance_filter_enabled:
        filters.append(LowChannelVarianceFilter(thresholds=variance_thresholds))

    filtered = apply_filters(ds, filters, num_proc=num_proc, use_cache=use_cache) if filters else ds

    user_ids = filtered["user_id"]
    dates = filtered["date"]

    kept_days: dict[str, list[str]] = defaultdict(list)
    for user_id, date in zip(user_ids, dates):
        kept_days[str(user_id)].append(str(date))

    # Ensure deterministic output for reproducible JSON artifacts.
    mask = {
        user_id: sorted(set(day_list))
        for user_id, day_list in sorted(kept_days.items(), key=lambda x: x[0])
    }

    logger.info(
        "Generated day-drop mask: %d users, %d kept days",
        len(mask),
        sum(len(v) for v in mask.values()),
    )
    return mask


def save_day_drop_mask(mask: dict[str, list[str]], output_path: str | Path) -> Path:
    """Persist day-drop mask as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mask, indent=2, ensure_ascii=False))
    logger.info("Saved day-drop mask to %s", output)
    return output


def read_day_drop_mask(file_path: str | Path) -> dict[str, list[str]]:
    """Load day-drop mask JSON from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    data = json.loads(path.read_text())
    return {str(k): [str(x) for x in v] for k, v in data.items()}


# def parse_args() -> argparse.Namespace:
#     """Parse CLI arguments."""
#     parser = argparse.ArgumentParser(description="Generate day-level keep-mask from daily_hf")
#     parser.add_argument("--daily-hf-dir", required=True, type=Path)
#     parser.add_argument("--output-json", required=True, type=Path)
#     parser.add_argument("--min-wear-fraction", type=float, default=0.5)
#     parser.add_argument("--disable-variance-filter", action="store_true")
#     parser.add_argument("--num-proc", type=int, default=1)
#     parser.add_argument("--use-cache", action="store_true")
#     return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # args = parse_args()

    # day_mask = generate_day_drop_mask(
    #     daily_hf_dir=args.daily_hf_dir,
    #     min_wear_fraction=args.min_wear_fraction,
    #     variance_filter_enabled=not args.disable_variance_filter,
    #     num_proc=args.num_proc,
    #     use_cache=args.use_cache,
    # )
    day_mask = generate_day_drop_mask(
        daily_hf_dir="/rds/general/user/lp925/home/code/MHC-benchmark/data/processed/daily_hf"
    )
    save_day_drop_mask(day_mask, "/rds/general/user/lp925/home/code/MHC-benchmark/data/forecasting_sample_index/new_day_drop_mask.json")