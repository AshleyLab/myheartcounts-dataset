"""Stream raw (gt, pred) pairs to per-channel Parquet files during imputation evaluation.

Writes masked (sample_idx, timestep, gt, pred) tuples batch-by-batch to per-channel
Parquet files (one file per channel), keeping memory constant. All metrics can be
reconstructed post-hoc from these pairs.

Per-channel file naming: ``pairs_ch{ch:02d}.parquet``

Schemas (compact types for minimal disk/memory):
  Continuous channels (0-6):
    - sample_idx: int32
    - timestep:   int16
    - gt:         float16
    - pred:       float16

  Binary channels (7-18):
    - sample_idx: int32
    - timestep:   int16
    - gt:         bool (bit-packed)
    - pred:       float16
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.hf_config import BINARY_CHANNEL_INDICES

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("sample_idx", pa.int32()),
        pa.field("user_id", pa.utf8()),
        pa.field("date", pa.utf8()),
    ]
)

CONTINUOUS_PAIR_SCHEMA = pa.schema(
    [
        pa.field("sample_idx", pa.int32()),
        pa.field("timestep", pa.int16()),
        pa.field("gt", pa.float16()),
        pa.field("pred", pa.float16()),
    ]
)

BINARY_PAIR_SCHEMA = pa.schema(
    [
        pa.field("sample_idx", pa.int32()),
        pa.field("timestep", pa.int16()),
        pa.field("gt", pa.bool_()),
        pa.field("pred", pa.float16()),
    ]
)

_BINARY_SET = set(BINARY_CHANNEL_INDICES)


def _channel_file(output_path: Path, ch: int) -> Path:
    return output_path / f"pairs_ch{ch:02d}.parquet"


class PairWriter:
    """Stream masked (gt, pred) pairs to per-channel Parquet files.

    One PairWriter per (scenario, split). Uses PyArrow for efficient columnar
    writes with compact schemas. Implements context manager protocol.

    Usage:
        with PairWriter(output_dir / "random_noise" / "test") as pw:
            pw.write_batch(ground_truth, imputed, masks, sample_indices)
    """

    def __init__(self, output_path: str | Path):
        """Initialize PairWriter.

        Args:
            output_path: Directory to write per-channel Parquet files into.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._total_rows = 0

    def _ensure_writer(self, ch: int) -> pq.ParquetWriter:
        if ch not in self._writers:
            schema = BINARY_PAIR_SCHEMA if ch in _BINARY_SET else CONTINUOUS_PAIR_SCHEMA
            self._writers[ch] = pq.ParquetWriter(
                _channel_file(self.output_path, ch),
                schema,
                compression="zstd",
            )
        return self._writers[ch]

    def write_batch(
        self,
        ground_truth: np.ndarray,
        imputed: np.ndarray,
        artificial_masks: np.ndarray,
        sample_indices: np.ndarray | list[int],
    ) -> None:
        """Extract and write masked (gt, pred) pairs from a batch.

        Args:
            ground_truth: Shape (N, C, T) — original unmasked values.
            imputed: Shape (N, C, T) — imputed values.
            artificial_masks: Shape (N, C, T) — binary, 1 = was artificially masked.
            sample_indices: Length-N array of split-local sample indices.
        """
        if len(ground_truth) == 0:
            return

        sample_indices = np.asarray(sample_indices, dtype=np.int32)
        n_samples, n_channels, n_timesteps = ground_truth.shape

        mask_bool = artificial_masks == 1

        for ch in range(n_channels):
            ch_mask = mask_bool[:, ch, :]  # (N, T)
            s_idx, t_idx = np.where(ch_mask)

            if len(s_idx) == 0:
                continue

            gt_vals = ground_truth[s_idx, ch, t_idx]
            pred_vals = imputed[s_idx, ch, t_idx]

            # Filter non-finite
            finite_mask = np.isfinite(gt_vals) & np.isfinite(pred_vals)
            if not np.all(finite_mask):
                s_idx = s_idx[finite_mask]
                t_idx = t_idx[finite_mask]
                gt_vals = gt_vals[finite_mask]
                pred_vals = pred_vals[finite_mask]

            if len(s_idx) == 0:
                continue

            mapped_sample_idx = sample_indices[s_idx]
            timestep_vals = t_idx.astype(np.int16)
            is_binary = ch in _BINARY_SET

            if is_binary:
                table = pa.table(
                    {
                        "sample_idx": pa.array(mapped_sample_idx, type=pa.int32()),
                        "timestep": pa.array(timestep_vals, type=pa.int16()),
                        "gt": pa.array(gt_vals > 0.5, type=pa.bool_()),
                        "pred": pa.array(pred_vals.astype(np.float16), type=pa.float16()),
                    },
                    schema=BINARY_PAIR_SCHEMA,
                )
            else:
                table = pa.table(
                    {
                        "sample_idx": pa.array(mapped_sample_idx, type=pa.int32()),
                        "timestep": pa.array(timestep_vals, type=pa.int16()),
                        "gt": pa.array(gt_vals.astype(np.float16), type=pa.float16()),
                        "pred": pa.array(pred_vals.astype(np.float16), type=pa.float16()),
                    },
                    schema=CONTINUOUS_PAIR_SCHEMA,
                )

            writer = self._ensure_writer(ch)
            writer.write_table(table)
            self._total_rows += len(mapped_sample_idx)

    def write_extracted_pairs(
        self,
        sample_idx: np.ndarray,
        channel: np.ndarray,
        timestep: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        """Write pre-extracted (gt, pred) pairs directly to per-channel Parquet files.

        Accepts column arrays that have already been extracted from masked positions
        (e.g. by ``_extract_pairs`` in the worker). Splits by channel and writes
        to the appropriate per-channel file with the correct schema.

        Args:
            sample_idx: (M,) int32 — split-local sample indices.
            channel: (M,) int8 — channel indices.
            timestep: (M,) int16 — timestep indices.
            gt: (M,) float32 — ground truth values.
            pred: (M,) float32 — predicted/imputed values.
        """
        if len(sample_idx) == 0:
            return

        channel = np.asarray(channel)
        for ch in np.unique(channel):
            ch_mask = channel == ch
            ch_sample_idx = sample_idx[ch_mask]
            ch_timestep = timestep[ch_mask].astype(np.int16)
            ch_gt = gt[ch_mask]
            ch_pred = pred[ch_mask]

            is_binary = int(ch) in _BINARY_SET

            if is_binary:
                table = pa.table(
                    {
                        "sample_idx": pa.array(ch_sample_idx, type=pa.int32()),
                        "timestep": pa.array(ch_timestep, type=pa.int16()),
                        "gt": pa.array(ch_gt > 0.5, type=pa.bool_()),
                        "pred": pa.array(ch_pred.astype(np.float16), type=pa.float16()),
                    },
                    schema=BINARY_PAIR_SCHEMA,
                )
            else:
                table = pa.table(
                    {
                        "sample_idx": pa.array(ch_sample_idx, type=pa.int32()),
                        "timestep": pa.array(ch_timestep, type=pa.int16()),
                        "gt": pa.array(ch_gt.astype(np.float16), type=pa.float16()),
                        "pred": pa.array(ch_pred.astype(np.float16), type=pa.float16()),
                    },
                    schema=CONTINUOUS_PAIR_SCHEMA,
                )

            writer = self._ensure_writer(int(ch))
            writer.write_table(table)
            self._total_rows += len(ch_sample_idx)

    def close(self) -> None:
        """Close all per-channel Parquet writers and drop page cache.

        Advises the kernel to evict each file's pages from page cache via
        ``posix_fadvise(DONTNEED)``. Under cgroups v1 (SLURM), page cache
        counts toward the memory limit, so dropping it prevents OOM when
        multiple methods run sequentially in the same job.
        """
        for ch, writer in self._writers.items():
            writer.close()
            # Drop page cache so it doesn't count against the cgroup memory limit
            fpath = _channel_file(self.output_path, ch)
            if not hasattr(os, "posix_fadvise"):
                continue  # cache hint unavailable on this platform (e.g. macOS)
            try:
                fd = os.open(str(fpath), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass  # Best-effort; non-fatal if unsupported
        self._writers.clear()
        if self._total_rows > 0:
            logger.info(
                f"Wrote {self._total_rows:,} pairs across per-channel files in {self.output_path}"
            )

    def __enter__(self) -> PairWriter:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing all writers."""
        self.close()

    @property
    def total_rows(self) -> int:
        """Total number of rows written so far across all channels."""
        return self._total_rows


def write_sample_manifest(
    pairs_dir: str | Path,
    split_indices: list[int],
    split_name: str,
    all_user_ids: list[str],
    all_dates: list[str],
) -> Path:
    """Write a sample manifest mapping split-local index to user_id and date.

    Saved as ``{pairs_dir}/manifest_{split_name}.parquet`` alongside
    ``channel_stds.npy``.  Enables post-hoc sensitivity analysis from
    saved pair files without replaying the data loading pipeline.

    Args:
        pairs_dir: Base pairs directory.
        split_indices: Global HF dataset indices for this split.
            Split-local index *i* maps to ``all_user_ids[split_indices[i]]``.
        split_name: Split name (e.g. ``"val"``, ``"test"``).
        all_user_ids: Full user_id column from the HF dataset.
        all_dates: Full date column from the HF dataset.

    Returns:
        Path to the written manifest file.
    """
    pairs_dir = Path(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(split_indices)
    sample_idx = np.arange(n_samples, dtype=np.int32)

    user_ids = [all_user_ids[gi] for gi in split_indices]
    dates = [all_dates[gi] for gi in split_indices]

    table = pa.table(
        {
            "sample_idx": pa.array(sample_idx, type=pa.int32()),
            "user_id": pa.array(user_ids, type=pa.utf8()),
            "date": pa.array(dates, type=pa.utf8()),
        },
        schema=MANIFEST_SCHEMA,
    )

    manifest_path = pairs_dir / f"manifest_{split_name}.parquet"
    pq.write_table(table, manifest_path, compression="zstd")
    logger.info(f"Wrote sample manifest ({n_samples} samples) to {manifest_path}")
    return manifest_path


def load_sample_manifest(
    pairs_dir: str | Path,
    split_name: str,
) -> pa.Table | None:
    """Load a sample manifest for a split.

    Args:
        pairs_dir: Base pairs directory containing manifest files.
        split_name: Split name (e.g. ``"val"``, ``"test"``).

    Returns:
        PyArrow table with columns ``(sample_idx, user_id, date)``,
        or ``None`` if the manifest file does not exist.
    """
    manifest_path = Path(pairs_dir) / f"manifest_{split_name}.parquet"
    if not manifest_path.exists():
        return None
    return pq.read_table(manifest_path)


# ---------------------------------------------------------------------------
# Fallback sidecar (model-capability counts, persisted alongside pairs)
# ---------------------------------------------------------------------------


def write_fallback_sidecar(
    pairs_dir: str | Path,
    split_name: str,
    per_scenario_counts: dict[str, dict],
) -> Path:
    """Persist per-(scenario, split) fallback + applicability counts.

    Writes ``{pairs_dir}/fallback_{split_name}.json``. The new sidecar
    replaces the in-memory ``MetricAccumulator`` counters: the harness
    streams everything to disk, and the canonical producer / aggregator
    reads it back to populate ``overall_fallback_rate`` /
    ``fallback_rate`` / ``n_total`` / ``n_applicable`` in the final
    :class:`ImputationResults.scenarios` dict.

    Args:
        pairs_dir: The pairs root (sibling of ``manifest_{split}.parquet``).
        split_name: Split name (e.g. ``"test"``).
        per_scenario_counts: ``{scenario: {n_applicable, n_total,
            fallback_substituted, fallback_asked}}`` where the two
            ``fallback_*`` entries are per-channel ``(C,)`` int64 arrays
            (or lists). Serialised as lists.

    Returns:
        Path to the written sidecar.
    """
    import json

    pairs_dir = Path(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for scenario, counts in per_scenario_counts.items():
        sub = counts.get("fallback_substituted", [])
        asked = counts.get("fallback_asked", [])
        if hasattr(sub, "tolist"):
            sub = sub.tolist()
        if hasattr(asked, "tolist"):
            asked = asked.tolist()
        out[scenario] = {
            "n_applicable": int(counts.get("n_applicable", 0)),
            "n_total": int(counts.get("n_total", 0)),
            "fallback_substituted": [int(x) for x in sub],
            "fallback_asked": [int(x) for x in asked],
        }
    path = pairs_dir / f"fallback_{split_name}.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote fallback sidecar (%d scenarios) to %s", len(out), path)
    return path


def read_fallback_sidecar(
    pairs_dir: str | Path,
    split_name: str,
) -> dict[str, dict] | None:
    """Read the fallback sidecar for a split.

    Returns ``None`` when the sidecar is missing (e.g. an older
    pre-Phase-A pairs dir). Callers populate fallback fields as
    ``0.0`` / ``None`` in that case to preserve the historical contract.
    """
    import json

    path = Path(pairs_dir) / f"fallback_{split_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def merge_counts_and_fallback(
    sidecar: dict | None,
    *,
    n_channels: int,
) -> dict:
    """Derive ``overall_fallback_rate`` / ``fallback_rate`` / ``n_*`` from a sidecar.

    Port of the arithmetic the old ``MetricAccumulator.compute`` did on
    its in-memory counters. Returns a dict with the four fields
    :class:`ImputationResults.scenarios` exposes today, ready to merge
    with the producer's display-metrics dict.

    Missing-sidecar fallback: zero counts, zero rates — matches what the
    legacy path emitted when ``fallback_fill`` was ``None``.
    """
    if sidecar is None:
        return {
            "n_applicable": 0,
            "n_total": 0,
            "overall_fallback_rate": 0.0,
            "fallback_rate": {f"ch_{ch}": 0.0 for ch in range(n_channels)},
        }
    n_applicable = int(sidecar.get("n_applicable", 0))
    n_total = int(sidecar.get("n_total", 0))
    sub = list(sidecar.get("fallback_substituted") or [])
    asked = list(sidecar.get("fallback_asked") or [])
    asked_total = sum(int(x) for x in asked)
    sub_total = sum(int(x) for x in sub)
    overall = (sub_total / asked_total) if asked_total > 0 else 0.0
    per_ch: dict[str, float] = {}
    for ch in range(n_channels):
        a = int(asked[ch]) if ch < len(asked) else 0
        s = int(sub[ch]) if ch < len(sub) else 0
        per_ch[f"ch_{ch}"] = (s / a) if a > 0 else 0.0
    return {
        "n_applicable": n_applicable,
        "n_total": n_total,
        "overall_fallback_rate": overall,
        "fallback_rate": per_ch,
    }
