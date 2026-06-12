"""Gate for the TSFM extraction → DataLoader migration (run once, then keep for audits).

Builds every user's hourly timeline both ways and byte-compares them:

  - OLD path (inline verbatim): raw ``daily_hourly_hf`` rows, sorted by date,
    ``ZeroToNaNTransform`` per (19, 24) day, laid on the continuous hourly grid.
  - NEW path: ``DataLoader.user_days`` (NaN-at-masked, date-ascending) through
    ``tsfm.build_user_timeline`` (zero-fills back before the same transform).

Timeline equality ⇒ window equality (``build_window`` is shared + deterministic),
so identical hashes prove the migration cannot change extracted embeddings.

Usage: MHC_DATA_DIR=... PYTHONPATH=src python scripts/validate_tsfm_loader_migration.py
"""

import hashlib

import numpy as np
import torch

HOURS_PER_DAY = 24
N_CHANNELS = 19


def _h(timeline) -> str:
    if timeline is None:
        return "EMPTY"
    h = hashlib.sha256()
    h.update(str(timeline.start_date).encode())
    h.update(timeline.values.tobytes())
    h.update(timeline.observed_hours.tobytes())
    return h.hexdigest()[:16]


def old_timelines():
    """Old path, replicated verbatim (bulk column reads for speed; same bytes)."""
    import datasets as hf_ds
    import pandas as pd

    from data.transforms.nan_transforms import ZeroToNaNTransform
    from downstream_evaluation.data.splits import load_split_file
    from downstream_evaluation.models.tsfm import UserTimeline
    from openmhc._evaluate import _DatasetPaths

    paths = _DatasetPaths.resolve(None)
    ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
    split_users = load_split_file(paths.splits_file)
    user_to_split = {str(u): s for s, us in split_users.items() for u in us}

    user_ids = np.asarray(ds["user_id"], dtype=object)
    values_all = np.asarray(ds["values"], dtype=np.float32)  # (N, 19, 24) raw zero-filled
    dates_all = np.asarray([str(d)[:10] for d in ds["date"]], dtype=object)
    del ds

    # _group_indices logic (the shipped daily_hourly_hf has no quality columns).
    grouped: dict = {}
    for idx, uid in enumerate(user_ids):
        uid = str(uid)
        if uid in user_to_split:
            grouped.setdefault(uid, []).append(idx)

    zero_to_nan = ZeroToNaNTransform()
    out: dict[str, str] = {}
    for uid, idxs in grouped.items():
        idxs = sorted(idxs, key=lambda i: dates_all[i])  # rows.sort(key=date)
        first = pd.Timestamp(dates_all[idxs[0]])
        last = pd.Timestamp(dates_all[idxs[-1]])
        n_days = int((last - first).days) + 1
        timeline = np.full((n_days * HOURS_PER_DAY, N_CHANNELS), np.nan, dtype=np.float32)
        for i in idxs:
            off = int((pd.Timestamp(dates_all[i]) - first).days)
            v = zero_to_nan(torch.from_numpy(values_all[i])).numpy()
            timeline[off * HOURS_PER_DAY : (off + 1) * HOURS_PER_DAY, :] = v.T
        observed = ~np.isnan(timeline).all(axis=1)
        out[uid] = _h(UserTimeline(start_date=first, values=timeline, observed_hours=observed))
    return out


def new_timelines(users):
    from downstream_evaluation.data.loader import DataLoader
    from downstream_evaluation.models.tsfm import build_user_timeline

    dl = DataLoader(None)
    out: dict[str, str] = {}
    for uid in users:
        day_values, day_dates = dl.user_days(uid)
        out[uid] = _h(build_user_timeline(day_values, day_dates, N_CHANNELS))
    return out


def main() -> None:
    old = old_timelines()
    print(f"old path: {len(old)} users", flush=True)
    new = new_timelines(sorted(old))
    same = sum(1 for u in old if old[u] == new[u])
    diff = [u for u in old if old[u] != new[u]]
    print(f"identical: {same}/{len(old)} users")
    if diff:
        print("DIFFERING users (first 10):", diff[:10])
        raise SystemExit(1)
    print("GATE PASSED: loader-based timelines are byte-identical to the raw-row path")


if __name__ == "__main__":
    main()
