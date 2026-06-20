"""Build per-user forecasting sample-index files from hourly trajectories."""

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime

import datasets as hf_ds
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SampleIndexGenerator:
    """Generate and persist candidate forecasting start-day indices by user."""

    def __init__(
        self,
        hourly_trajectory_path: str,
        sample_index_path: str,
        day_remain_mask_path: str,
        filter_parameters: dict,
        forecasting_length: int = 24,
    ):
        """Initialize sample-index generation inputs and validation state.

        Args:
            hourly_trajectory_path: HuggingFace hourly trajectory dataset path.
            sample_index_path: Base output path used to save generated index JSON files.
            day_remain_mask_path: JSON path for day-level retain mask keyed by user.
            filter_parameters: Dict describing optional filtering stages and parameters.
            forecasting_length: Forecast horizon in hours used to bound valid start days.
        """
        self.hourly_trajectory_path = hourly_trajectory_path
        self.sample_index_path = sample_index_path
        self.day_remain_mask_path = day_remain_mask_path
        self.filter_parameters = filter_parameters
        self.forecasting_length = int(forecasting_length)

        if self.forecasting_length <= 0:
            raise ValueError("forecasting_length must be a positive integer")

        self.sample_index = {}

        self._check_all_data()

        logger.info("Loading hourly trajectory data from %s", hourly_trajectory_path)
        logger.info("Will saving generated sample index to %s", sample_index_path)
        logger.info("Loading day remain mask from %s", day_remain_mask_path)
        logger.info("Initialized SampleIndexGenerator with parameters: %s", filter_parameters)
        logger.info("Forecasting length (hours): %d", self.forecasting_length)

    def _check_all_data(self):
        if not os.path.exists(self.hourly_trajectory_path):
            raise FileNotFoundError(
                f"Hourly trajectory dataset not found at {self.hourly_trajectory_path}"
            )

        if not os.path.exists(self.day_remain_mask_path):
            raise FileNotFoundError(
                f"Day remain mask file not found at {self.day_remain_mask_path}"
            )
        else:
            with open(self.day_remain_mask_path, encoding="utf-8") as f:
                self.day_remain_mask = json.load(f)

    def _load_sample_index(self, name):
        # Read the parent directory of self.sample_index_path.
        output_dir = os.path.dirname(self.sample_index_path)

        # Build output filename as name + .json.
        filename = f"{name}.json" if not str(name).endswith(".json") else str(name)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                self.sample_index = json.load(f)
            return True
        return False

    def _save_sample_index(self, name):
        # Before saving/stats, drop users with empty index lists.
        self.sample_index = {
            user_id: indices for user_id, indices in self.sample_index.items() if len(indices) > 0
        }

        # Read the parent directory of self.sample_index_path.
        output_dir = os.path.dirname(self.sample_index_path)

        # Build output filename as name + .json.
        filename = f"{name}.json" if not str(name).endswith(".json") else str(name)
        output_path = os.path.join(output_dir, filename)

        # Save self.sample_index to the computed output path.
        if not os.path.exists(self.sample_index_path):
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_index, f, ensure_ascii=False, indent=2)

            logger.info("Saved sample index to %s", output_path)

        logger.info("****Sample index stats for %s:****", name)

        # Log user count, total sample count, and per-user quantiles (10%..90%).
        user_count = len(self.sample_index)
        per_user_counts = np.array([len(v) for v in self.sample_index.values()], dtype=np.int64)
        total_samples = int(per_user_counts.sum()) if per_user_counts.size > 0 else 0

        logger.info("Sample index stats: users=%d, total_samples=%d", user_count, total_samples)

        if per_user_counts.size > 0:
            quantile_levels = np.arange(0.1, 1.0, 0.1)
            quantile_values = np.quantile(per_user_counts, quantile_levels)
            quantile_pairs = ", ".join(
                [f"q{int(q * 100)}={int(v)}" for q, v in zip(quantile_levels, quantile_values)]
            )
            logger.info("Per-user sample count quantiles: %s", quantile_pairs)
        else:
            logger.info("Per-user sample count quantiles: empty sample index")

    def _missing_mask_filter(self, ds, sample_index):
        filtered_sample_index: dict[str, list[int]] = {user_id: [] for user_id in sample_index}
        forecast_day_count = math.ceil(self.forecasting_length / 24)

        # Stream records and filter on the fly to avoid loading a full map in memory.
        for example in ds:
            user_id = str(example["user_id"])
            if user_id not in sample_index:
                continue

            indices = sample_index[user_id]
            timestamps = example["timestamps"]
            allowed_days = set(self.day_remain_mask.get(user_id, []))

            kept_indices: list[int] = []
            for i in indices:
                target_day_idx = int(i)
                all_future_days_valid = True

                for day_offset in range(forecast_day_count):
                    day_idx = target_day_idx + day_offset
                    target_hour_idx = day_idx * 24
                    if target_hour_idx >= len(timestamps):
                        all_future_days_valid = False
                        break

                    ts = timestamps[target_hour_idx]
                    day_str = str(ts).split("T", 1)[0]

                    if day_str not in allowed_days:
                        all_future_days_valid = False
                        break

                if all_future_days_valid:
                    kept_indices.append(i)

            filtered_sample_index[user_id] = kept_indices
        return filtered_sample_index

    def _historical_check_filter(self, ds, sample_index, parameters):
        filtered_sample_index: dict[str, list[int]] = {user_id: [] for user_id in sample_index}

        recent_day_count = int(parameters["recent_day_count"])
        minimum_valid_day = int(parameters["minimum_valid_day"])

        for example in ds:
            user_id = str(example["user_id"])
            if user_id not in sample_index:
                continue

            indices = sample_index[user_id]
            timestamps = example["timestamps"]
            allowed_days = set(self.day_remain_mask.get(user_id, []))

            kept_indices: list[int] = []

            for i in indices:
                target_day_idx = int(i)

                history_start_day = max(0, target_day_idx - recent_day_count)
                valid_day_count = 0

                for day_idx in range(history_start_day, target_day_idx):
                    ts_idx = day_idx * 24
                    if ts_idx >= len(timestamps):
                        continue

                    ts = timestamps[ts_idx]
                    day_str = str(ts).split("T", 1)[0]

                    if day_str in allowed_days:
                        valid_day_count += 1

                    if valid_day_count >= minimum_valid_day:
                        kept_indices.append(i)
                        break

            filtered_sample_index[user_id] = kept_indices

        return filtered_sample_index

    def _maximum_sample_count_filter(self, sample_index, max_count):
        filtered_sample_index: dict[str, list[int]] = {}
        for user_id, indices in sample_index.items():
            if len(indices) <= max_count:
                filtered_sample_index[user_id] = indices
            else:
                filtered_sample_index[user_id] = random.sample(indices, max_count)
        return filtered_sample_index

    def _filter_name_suffix(self, key, value):
        if key == "missing_mask":
            return "_M"

        if key == "historical_check":
            return f"_H_{value['recent_day_count']}_{value['minimum_valid_day']}"

        if key == "maximum_sample_count_per_user":
            return f"_S_{value}"

    def generate(self):
        """Generate sample indices, apply optional filters, and persist outputs.

        Returns:
            Dict mapping user IDs to retained forecast start-day indices.
        """
        if os.path.exists(self.sample_index_path):
            logger.info(
                "Sample index file already exists at %s; loading and returning it.",
                self.sample_index_path,
            )
            with open(self.sample_index_path, encoding="utf-8") as f:
                self.sample_index = json.load(f)
            return self.sample_index

        # Load hourly trajectory dataset
        logger.info(
            "Loading hourly trajectory Hugging Face dataset from %s", self.hourly_trajectory_path
        )
        ds = hf_ds.load_from_disk(self.hourly_trajectory_path)
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))

        logger.info("Loaded dataset with %d user trajectories", len(ds))

        # Generate sample index in format:
        # {
        #   "user_id": [1, 2, 3, ...],
        #   ...
        # }
        if not self._load_sample_index(f"sample_index_P_{self.forecasting_length}_raw"):
            raw_sample_index: dict[str, list[int]] = {}
            for row_idx, example in enumerate(tqdm(ds, desc="Generating sample index")):
                values = np.asarray(example["values"], dtype=np.float32)
                user_id = str(example["user_id"])

                total_hours = int(values.shape[0])
                max_start_day = (total_hours - self.forecasting_length) // 24
                candidate_indices = list(range(1, max_start_day + 1)) if max_start_day >= 1 else []

                raw_sample_index[user_id] = candidate_indices

            self.sample_index = raw_sample_index
        self._save_sample_index(f"sample_index_P_{self.forecasting_length}_raw")

        # implement filtering logic based on self.filter_parameters
        if self.filter_parameters:
            sample_index_name = f"sample_index_P_{self.forecasting_length}"
            for key, value in self.filter_parameters.items():
                logger.info("Applying filter: %s = %s", key, value)
                sample_index_name += self._filter_name_suffix(key, value)

                if key == "missing_mask":
                    if not self._load_sample_index(sample_index_name):
                        self.sample_index = self._missing_mask_filter(ds, self.sample_index.copy())
                    self._save_sample_index(sample_index_name)

                if key == "historical_check":
                    if not self._load_sample_index(sample_index_name):
                        self.sample_index = self._historical_check_filter(
                            ds, self.sample_index.copy(), value
                        )
                    self._save_sample_index(sample_index_name)

                if key == "maximum_sample_count_per_user":
                    if not self._load_sample_index(sample_index_name):
                        self.sample_index = self._maximum_sample_count_filter(
                            self.sample_index.copy(), value
                        )
                    self._save_sample_index(sample_index_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self._save_sample_index(f"sample_index_final_{timestamp}")
        return self.sample_index


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Generate forecasting sample index with configurable filters."
    )
    parser.add_argument("--hourly_trajectory_path", type=str, required=True)
    parser.add_argument("--sample_index_path", type=str, required=True)
    parser.add_argument("--day_remain_mask_path", type=str, required=True)
    parser.add_argument(
        "--forecasting_length",
        type=int,
        default=24,
        help="Forecast horizon in hours (e.g. 24 or 48).",
    )
    parser.add_argument(
        "--filter_parameters_json",
        type=str,
        default=None,
        help="JSON string for filter parameters, e.g. '{\"missing_mask\": true}'.",
    )
    parser.add_argument(
        "--filter_parameters_path",
        type=str,
        default=None,
        help="Path to JSON file containing filter parameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    return parser


def _load_filter_parameters(filter_parameters_json=None, filter_parameters_path=None):
    if filter_parameters_json and filter_parameters_path:
        raise ValueError(
            "Please provide only one of --filter_parameters_json or --filter_parameters_path"
        )

    if filter_parameters_json:
        return json.loads(filter_parameters_json)

    if filter_parameters_path:
        with open(filter_parameters_path, encoding="utf-8") as f:
            return json.load(f)

    return {}


def main():
    """CLI entry point for generating forecasting sample-index artifacts."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    filter_parameters = _load_filter_parameters(
        filter_parameters_json=args.filter_parameters_json,
        filter_parameters_path=args.filter_parameters_path,
    )

    generator = SampleIndexGenerator(
        hourly_trajectory_path=args.hourly_trajectory_path,
        sample_index_path=args.sample_index_path,
        day_remain_mask_path=args.day_remain_mask_path,
        filter_parameters=filter_parameters,
        forecasting_length=args.forecasting_length,
    )
    generator.generate()


if __name__ == "__main__":
    main()
