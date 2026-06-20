"""Demographic subgroup mapping for sensitivity analysis.

Builds per-sample demographic attributes (age group, biological sex) from
the HuggingFace dataset metadata and the Labels API, enabling subgroup-level
metric breakdowns in imputation evaluation.
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from labels.api import STORE, LabelsStore, years_between_birth_year

logger = logging.getLogger(__name__)


def bin_age(age: int, age_bins: list[int]) -> str:
    """Bin an age into a decade-style label.

    Args:
        age: Age in years.
        age_bins: Sorted list of bin edges, e.g. [18, 30, 40, 50, 60].

    Returns:
        Bin label, e.g. "30-39" or "60+".
    """
    for i in range(len(age_bins) - 1):
        if age_bins[i] <= age < age_bins[i + 1]:
            return f"{age_bins[i]}-{age_bins[i + 1] - 1}"
    if age >= age_bins[-1]:
        return f"{age_bins[-1]}+"
    return "unknown"


def get_user_demographics(
    store: LabelsStore,
    user_ids: list[str],
) -> dict[str, dict[str, object]]:
    """Look up birth year and biological sex for each user.

    Args:
        store: Labels store for data access.
        user_ids: Unique user identifiers (health codes).

    Returns:
        Dict mapping user_id -> {"birth_year": int | None, "sex": str}.
        Sex is "male", "female", or "unknown".
    """
    demographics: dict[str, dict[str, object]] = {}

    for uid in user_ids:
        birth_year = None
        sex = "unknown"

        try:
            birth_year = store.enrollment.get_birth_year(uid)
        except KeyError:
            pass

        try:
            series = store.labels_index.get_series("BiologicalSex", uid)
            result = series.nearest(pd.Timestamp("2020-01-01"))
            raw_value = result.value
            if isinstance(raw_value, bool):
                sex = "male" if raw_value else "female"
            elif isinstance(raw_value, str):
                lower = raw_value.lower().strip()
                if lower == "male":
                    sex = "male"
                elif lower == "female":
                    sex = "female"
        except (KeyError, LookupError):
            pass

        demographics[uid] = {"birth_year": birth_year, "sex": sex}

    # Surface the case where age fairness will be silently empty downstream:
    # if the enrollment payload omits ``birth_year`` for every user, the age
    # subgroup map collapses to "unknown" and the Phase 2 aggregator drops
    # those rows from fairness_summary_bootstrap.csv with no error. The cli
    # log line is the only way an operator notices unless we warn here.
    if user_ids and all(v["birth_year"] is None for v in demographics.values()):
        logger.warning(
            "Age fairness will be empty: 0/%d users have birth_year. Check "
            "that the enrollment_info.json payload includes a 'birth_year' "
            "field (the canonical Dataverse schema). Sex fairness still "
            "works if BiologicalSex labels are present.",
            len(user_ids),
        )

    return demographics


def build_subgroup_mapping(
    hf_dataset,
    split_indices: dict[str, list[int]],
    age_bins: list[int],
    store: LabelsStore | None = None,
) -> dict[str, dict[int, dict[str, str]]]:
    """Build per-sample subgroup mapping for sensitivity analysis.

    For each sample in val/test splits, maps (split_local_index) to its
    demographic attributes (age_group, sex).

    Args:
        hf_dataset: HuggingFace dataset with user_id and date columns.
        split_indices: Maps split name -> list of global HF dataset indices.
        age_bins: Sorted list of age bin edges, e.g. [18, 30, 40, 50, 60].
        store: Optional LabelsStore instance (uses global STORE if None).

    Returns:
        Dict: {split_name: {split_local_idx: {"age_group": "30-39", "sex": "male"}}}.
        Only includes val and test splits.
    """
    if store is None:
        store = STORE

    # Pre-extract lightweight metadata columns once (avoids reading the 107 KB
    # "values" array per row, which under cgroups v1 faults ~46 GB of Arrow
    # page cache into memory for ~427 K samples).
    all_user_ids = list(hf_dataset["user_id"])
    all_dates = list(hf_dataset["date"])

    # Collect unique user IDs across val+test splits
    eval_user_ids: set[str] = set()
    for split_name in ("val", "test"):
        if split_name in split_indices:
            for idx in split_indices[split_name]:
                eval_user_ids.add(all_user_ids[idx])

    unique_users = sorted(eval_user_ids)
    logger.info(f"Looking up demographics for {len(unique_users)} unique users...")
    user_demographics = get_user_demographics(store, unique_users)

    # Build per-split mapping
    mapping: dict[str, dict[int, dict[str, str]]] = {}

    for split_name in ("val", "test"):
        if split_name not in split_indices:
            continue

        indices = split_indices[split_name]
        split_mapping: dict[int, dict[str, str]] = {}

        for local_idx, global_idx in enumerate(indices):
            user_id = all_user_ids[global_idx]
            date_str = all_dates[global_idx]

            demo = user_demographics.get(user_id, {"birth_year": None, "sex": "unknown"})

            # Compute age group
            age_group = "unknown"
            birth_year = demo["birth_year"]
            if birth_year is not None:
                try:
                    sample_date = pd.Timestamp(date_str)
                    age = years_between_birth_year(birth_year, sample_date)
                    age_group = bin_age(age, age_bins)
                except Exception:
                    pass

            split_mapping[local_idx] = {
                "age_group": age_group,
                "sex": demo["sex"],
            }

        mapping[split_name] = split_mapping

    return mapping


def format_subgroup_summary(mapping: dict[str, dict[int, dict[str, str]]]) -> str:
    """Format a human-readable summary of subgroup distributions.

    Args:
        mapping: Subgroup mapping as returned by build_subgroup_mapping().

    Returns:
        Multi-line string summarizing counts per attribute per group per split.
    """
    lines = ["Subgroup distributions:"]

    for split_name, split_mapping in sorted(mapping.items()):
        lines.append(f"\n  {split_name} ({len(split_mapping)} samples):")

        # Gather all attribute names from first entry
        if not split_mapping:
            lines.append("    (empty)")
            continue

        attrs = list(next(iter(split_mapping.values())).keys())

        for attr in attrs:
            counts = Counter(demo[attr] for demo in split_mapping.values())
            lines.append(f"    {attr}:")
            for group, count in sorted(counts.items()):
                pct = 100.0 * count / len(split_mapping)
                lines.append(f"      {group}: {count} ({pct:.1f}%)")

    return "\n".join(lines)
