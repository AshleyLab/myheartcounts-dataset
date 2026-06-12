"""Gate for the WBM weekly extraction → DataLoader migration.

The weekly windows are assembled by the *shared* ``IndexedWeekDataset`` class; the
migration only swaps its row source (raw ``daily_hourly_hf`` rows → the loader's
``as_daily_rows()`` view). Two checks:

  1. Row-source equivalence (decisive, all rows, vectorized): the adapter's
     zero-filled values + mask must byte-match the raw rows after the (19,24)→(24,19)
     transpose ``IndexedWeekDataset`` applies.
  2. End-to-end spot check: assemble every k-th window through both stacks and
     byte-compare the full sample dicts (values, mask, metadata).

Row equality + shared assembly code ⇒ identical windows ⇒ identical embeddings.

Usage: MHC_DATA_DIR=... PYTHONPATH=src python scripts/validate_wbm_loader_migration.py
"""

import hashlib

import numpy as np

SPOT_EVERY = 200  # end-to-end check every k-th window


def main() -> None:
    import datasets as hf_ds

    from data.datasets.indexed_week_dataset import IndexedWeekDataset
    from data.processing.build_window_index import load_window_index
    from openmhc._evaluate import _DatasetPaths

    from downstream_evaluation.data.loader import DataLoader

    paths = _DatasetPaths.resolve(None)

    # --- old row source: raw dataset, exactly as load_indexed_week_dataset opens it
    ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
    if isinstance(ds, hf_ds.DatasetDict):
        ds = hf_ds.concatenate_datasets(list(ds.values()))
    raw_values = np.asarray(ds["values"], dtype=np.float32)  # (N, 19, 24)
    raw_mask = np.asarray(ds["mask"], dtype=np.float32)

    # --- new row source: the loader's raw-form view
    dl = DataLoader(None)
    rows = dl.as_daily_rows()
    assert len(rows) == len(raw_values), f"row count {len(rows)} != {len(raw_values)}"

    new_values = np.nan_to_num(dl._values, nan=0.0)  # (N, 24, 19), what the adapter serves
    ok_v = np.array_equal(raw_values.transpose(0, 2, 1), new_values)
    ok_m = np.array_equal(raw_mask.transpose(0, 2, 1), dl._mask)
    print(f"row source: values {'IDENTICAL' if ok_v else 'DIFFER'},"
          f" mask {'IDENTICAL' if ok_m else 'DIFFER'} ({len(rows)} rows)", flush=True)
    if not (ok_v and ok_m):
        raise SystemExit(1)

    # --- end-to-end: same windows through both stacks
    window_index = load_window_index(str(paths.window_index))
    old_ds = IndexedWeekDataset(ds, window_index, window_size=7)
    new_ds = IndexedWeekDataset(rows, window_index, window_size=7)
    assert len(old_ds) == len(new_ds)

    def h(sample: dict) -> str:
        m = hashlib.sha256()
        for k in sorted(sample):
            v = sample[k]
            m.update(k.encode())
            m.update(v.tobytes() if isinstance(v, np.ndarray) else str(v).encode())
        return m.hexdigest()

    checked, bad = 0, []
    for i in range(0, len(old_ds), SPOT_EVERY):
        if h(old_ds[i]) != h(new_ds[i]):
            bad.append(i)
        checked += 1
    print(f"end-to-end: {checked - len(bad)}/{checked} windows identical")
    if bad:
        print("DIFFERING window indices (first 10):", bad[:10])
        raise SystemExit(1)
    print("GATE PASSED: loader-backed weekly windows are byte-identical to the raw-row path")


if __name__ == "__main__":
    main()
