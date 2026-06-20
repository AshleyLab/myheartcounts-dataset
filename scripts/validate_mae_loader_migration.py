"""Gate for the MAE extraction → DataLoader migration (run once, then keep for audits).

The migration replaced MAE's own ``daily_hf`` read + split/wear/variance filter with
(a) eligibility from the provider's daily lookup and (b) value fetch via the shared
minute-resolution ``DataLoader``. This gate proves the encoder sees byte-identical
inputs either way:

  - OLD path: split+wear+variance filter over ``daily_hf`` → kept ``(uid, date)`` → values.
  - NEW path: daily lookup ∩ cohort → ``DataLoader.participant_minute`` → ``(uid, date)`` → values.

Identical ``(uid, date)`` set + per-cell value bytes ⇒ identical day-embeddings (the
encoder is deterministic), so the cache's ``(user,date) → embedding`` mapping — hence
every downstream eval result — is unchanged. MAE pools per ``(user,date)`` from a dict,
so row order is irrelevant. No GPU required.

Usage: MHC_DATA_DIR=... PYTHONPATH=src python scripts/validate_mae_loader_migration.py
"""

import hashlib
from pathlib import Path

import datasets as hf_ds
import numpy as np
import pandas as pd

from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS
from downstream_evaluation.data.loader import DataLoader
from downstream_evaluation.data.provider import lookup_filename
from downstream_evaluation.data.splits import load_split_file
from openmhc._evaluate import _DatasetPaths

MAX_NONWEAR_MINUTES = 720
SAMPLE_USERS = 30


def _sha(a) -> bytes:
    return hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes()).digest()


paths = _DatasetPaths.resolve(None)
split_users = load_split_file(paths.splits_file)
cohort: set[str] = set()
for us in split_users.values():
    cohort |= {str(u) for u in us}

# --- OLD path: replicate the verbatim split+wear+variance filter over daily_hf ---
ds = hf_ds.load_from_disk(str(paths.daily_hf))
n = len(ds)
uid_arr = np.asarray(ds["user_id"], dtype=object).astype(str)
date_arr = np.asarray([str(d)[:10] for d in ds["date"]], dtype=object)
nonwear = np.asarray(ds["total_nonwear_minutes"], dtype=np.float64)
var = np.asarray(ds["channel_variance"], dtype=np.float64)
split_mask = np.array([u in cohort for u in uid_arr], dtype=bool)
wear_mask = nonwear <= MAX_NONWEAR_MINUTES
var_mask = np.ones(n, dtype=bool)
for ch, th in DEFAULT_VARIANCE_THRESHOLDS.items():
    if ch < var.shape[1]:
        col = var[:, ch]
        var_mask &= np.isnan(col) | (col >= th)
kept = np.where(split_mask & wear_mask & var_mask)[0]
old_set = {(uid_arr[i], date_arr[i]) for i in kept}
old_idx = {(uid_arr[i], date_arr[i]): int(i) for i in kept}
ds_vals = ds.select_columns(["values"])

# --- NEW path: eligibility = daily lookup ∩ cohort ---
lk = pd.read_parquet(
    Path(paths.root) / "processed" / lookup_filename("daily", full_history=True),
    columns=["user_id", "date"],
)
lk["user_id"] = lk["user_id"].astype(str)
lk["date"] = lk["date"].astype(str).str[:10]
new_by_user: dict[str, list[str]] = {}
for u, d in zip(lk["user_id"], lk["date"]):
    if u in cohort:
        new_by_user.setdefault(u, []).append(d)
new_set = {(u, d) for u, ds_ in new_by_user.items() for d in ds_}

print(f"OLD kept (filter): {len(old_set)}   NEW eligible (lookup ∩ cohort): {len(new_set)}")
print(f"(uid,date) set equal: {old_set == new_set}   "
      f"old-new: {len(old_set - new_set)}   new-old: {len(new_set - old_set)}")

# --- per-cell value bytes, sample of users via the loader ---
dl = DataLoader(None, resolution="minute")
sample = sorted(new_by_user)[:SAMPLE_USERS]
tot = ok = 0
bad = []
for u in sample:
    vals, used = dl.participant_minute(u, new_by_user[u])
    for v, d in zip(vals, used):
        tot += 1
        i = old_idx.get((u, d))
        if i is None:
            bad.append((u, d, "missing-in-old"))
        elif _sha(v) == _sha(ds_vals[i]["values"]):
            ok += 1
        else:
            bad.append((u, d, "byte-diff"))

print(f"sample users: {len(sample)}   days compared: {tot}   byte-identical: {ok}   bad: {len(bad)}")
if bad[:5]:
    print("  examples:", bad[:5])
gate = old_set == new_set and ok == tot and tot > 0
print(f"\nGATE {'PASSED' if gate else 'FAILED'}: migrated MAE extraction feeds "
      f"{'byte-identical inputs to the encoder' if gate else 'DIFFERENT inputs — investigate'}")
