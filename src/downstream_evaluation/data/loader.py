"""DataLoader — the single per-participant segment materializer.

Loads the row-aligned segment source **once**, indexes it by ``(user_id, date)``, and
materializes each participant's eligible segments on demand — per-(task, split) cohort
binding (``bind``) and whole-history access (``user_segments`` / ``segment_store``)
share the one read of ``daily_hourly_hf``.

Keyed by ``(user_id, date)`` (not lookup row positions), so the same selection works
regardless of which segment source / resolution backs it — no lookup↔segment
row-alignment is assumed.

The data handed to a model is **raw** — values with NaN at missing positions plus
the missingness mask. Normalization is the model's concern: encoders z-score with
train-split stats; the Linear baseline pools raw values directly.

See the "Data layer" section of ``src/downstream_evaluation/README.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from downstream_evaluation.data.provider import TaskData

logger = logging.getLogger(__name__)

# Channels and hours per day in daily_hourly_hf segments.
_N_CHANNELS = 19
_HOURS_PER_DAY = 24


@dataclass
class ParticipantSegments:
    """One participant's eligible segments (raw; ``values`` carry NaN at missing).

    Attributes:
        values: ``(n_segments, T, 19)`` sensor values, NaN where missing.
        mask: ``(n_segments, T, 19)`` missingness mask (1 = missing, 0 = observed).
    """

    values: np.ndarray
    mask: np.ndarray

    def as_array(self) -> np.ndarray:
        """The public ``(n_segments, T, 38)`` form handed to a :class:`~openmhc.Method`:
        values (channels 0-18, NaN at missing) concatenated with the missingness mask
        (channels 19-37). Both halves are already float32, so this is lossless."""
        return np.concatenate(
            [
                np.asarray(self.values, dtype=np.float32),
                np.asarray(self.mask, dtype=np.float32),
            ],
            axis=-1,
        )


def _day(d) -> str:
    """Normalize a date value to a ``YYYY-MM-DD`` key."""
    s = d if isinstance(d, str) else str(d)
    return s[:10]


def prepare_daily_hourly_hf(ds) -> "hf_ds.Dataset":
    """Convert a daily_hourly_hf dataset to the downstream-pipeline format.

    daily_hourly_hf stores values/mask as (19, 24) channels-first, zero-filled
    (no NaN in values, separate mask column).  The downstream pipeline expects
    (24, 19) time-first with NaN where mask==1.

    This function applies:
      1. Transpose values/mask from (C, H) = (19, 24) → (H, C) = (24, 19)
      2. Restore NaN in values where mask == 1 (so nanmean/nanstd work correctly)

    Note: HF Dataset.map() cannot change the Array2D shape declared in the
    schema, so we rebuild the dataset from a generator with the correct
    (24, 19) features schema.

    Args:
        ds: HuggingFace Dataset from daily_hourly_hf (each row has
            ``values`` (19, 24) and ``mask`` (19, 24)).

    Returns:
        Transformed HuggingFace Dataset with ``values`` (24, 19) containing
        NaN for missing positions and ``mask`` (24, 19).
    """
    import datasets as hf_ds

    logger.info(
        "Preparing daily_hourly_hf: transposing (19,24)->(24,19) and restoring NaN "
        "(%d samples)", len(ds),
    )

    # Bulk-read values and mask, transpose, restore NaN
    all_vals = np.asarray(ds["values"], dtype=np.float32)  # (N, 19, 24)
    all_mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 19, 24)

    # Transpose: (N, 19, 24) → (N, 24, 19)
    all_vals = np.ascontiguousarray(all_vals.transpose(0, 2, 1))
    all_mask = np.ascontiguousarray(all_mask.transpose(0, 2, 1))

    # Restore NaN where mask == 1 (missing) so nanmean/nanstd work
    all_vals[all_mask > 0.5] = np.nan

    # Read metadata columns
    user_ids = ds["user_id"]
    dates = ds["date"]
    n_valid_hours = ds["n_valid_hours"]

    # Build new dataset with corrected schema (24, 19) instead of (19, 24)
    new_features = hf_ds.Features({
        "values": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
        "mask": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
        "user_id": hf_ds.Value("string"),
        "date": hf_ds.Value("string"),
        "n_valid_hours": hf_ds.Value("int32"),
    })

    new_ds = hf_ds.Dataset.from_dict(
        {
            "values": all_vals,
            "mask": all_mask,
            "user_id": user_ids,
            "date": dates,
            "n_valid_hours": n_valid_hours,
        },
        features=new_features,
    )

    logger.info(
        "Prepared: %d samples, values shape=%s",
        len(new_ds),
        np.asarray(new_ds[0]["values"]).shape,
    )
    return new_ds


class DataLoader:
    """Load the segment source once; materialize per-participant segments by date."""

    def __init__(self, data_dir: str | None, granularity: str = "daily") -> None:
        """Args:
        data_dir: dataset root (``MHC_DATA_DIR`` / openmhc cache if ``None``).
        granularity: only ``"daily"`` is supported (hourly daily segments).
        """
        if granularity != "daily":
            raise NotImplementedError(
                f"DataLoader supports 'daily' granularity for now, got {granularity!r}"
            )
        self.granularity = granularity

        import datasets as hf_ds

        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.resolve(data_dir)
        ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))
        ds = prepare_daily_hourly_hf(ds)  # (24,19) time-first, NaN at missing

        # Keep as numpy for fast select-by-index.
        self._values = np.asarray(ds["values"], dtype=np.float32)  # (N, 24, 19)
        self._mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 24, 19)
        self._users = np.asarray(ds["user_id"], dtype=object).astype(str)  # per-row user_id
        dates = np.asarray(ds["date"], dtype=object)

        # (user_id, date) -> row, and user_id -> sorted row list (for whole-history
        # consumers). Built once; reused across all tasks/splits.
        self._row_by_key: dict[tuple[str, str], int] = {}
        self._rows_by_user: dict[str, list[int]] = {}
        for i, (u, d) in enumerate(zip(self._users, dates)):
            key = (u, _day(d))
            self._row_by_key[key] = i
            self._rows_by_user.setdefault(u, []).append(i)

    def participant(self, user_id, dates) -> ParticipantSegments:
        """One participant's eligible segments, selected by ``dates`` (date-ascending)."""
        u = str(user_id)
        rows = sorted(
            self._row_by_key[(u, k)] for d in dates if (u, k := _day(d)) in self._row_by_key
        )
        idx = np.asarray(rows, dtype=np.int64)
        return ParticipantSegments(values=self._values[idx], mask=self._mask[idx])

    def bind(self, td: TaskData) -> TaskData:
        """Fill ``td.inputs`` with one :class:`ParticipantSegments` per cohort user,
        selecting each user's eligible-date segments (``td.dates``)."""
        td.inputs = [self.participant(u, d) for u, d in zip(td.user_ids, td.dates)]
        return td

    # ----- whole-history access (for global-fit consumers: gru_d, multirocket) -----
    def user_segments(self, user_id) -> np.ndarray:
        """All of a participant's segments ``(n, 24, 19)`` (raw, NaN at missing)."""
        idx = np.asarray(sorted(self._rows_by_user.get(str(user_id), [])), dtype=np.int64)
        return self._values[idx]

    def segment_store(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The raw row-aligned store: ``(values, mask, users)`` — ``values`` / ``mask``
        are ``(N, 24, 19)`` (NaN at missing), ``users`` the per-row ``user_id`` array.

        For consumers that fit on the global train split or transform every segment
        (GRU-D, MultiRocket) and need the whole store, not just a cohort.
        """
        return self._values, self._mask, self._users
