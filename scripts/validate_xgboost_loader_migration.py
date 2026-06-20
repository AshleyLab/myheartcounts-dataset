"""Gate for the XGBoost selection → provider-eligibility migration (run once, keep for audits).

XGBoost keeps its vectorized-polars byte read (the documented exception); only its
eligibility moves to the provider's lookup. This gate runs the REAL builders on a sample
of daily_hf shards both ways and confirms byte-identical output:

  - OLD: re-derived split-free wear(≤720) + variance + (no) cutoff filter.
  - NEW: filter to the daily lookup's eligible (user,date) set.

The global row-set equivalence is already proven (selection proof: X == L), so this just
catches implementation bugs in the new filter (date matching, is_in / pyarrow mask). No
2.5-hour rebuild — 5 shards, both pipelines that read daily_hf.

Usage: MHC_DATA_DIR=... PYTHONPATH=src python scripts/validate_xgboost_loader_migration.py
"""

import glob
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from downstream_evaluation.data.provider import lookup_filename
from downstream_evaluation.models.xgboost.pipeline_curve_analysis import _accumulate_arrow_file
from downstream_evaluation.models.xgboost.pipeline_timeseries import build_user_features_chunked
from openmhc._evaluate import _DatasetPaths

MAX_NONWEAR = 720
N_SHARDS = 5


def col_equal(a, b) -> bool:
    try:
        return np.array_equal(a.astype(float), b.astype(float), equal_nan=True)
    except (ValueError, TypeError):
        return np.array_equal(a, b)


def frames_equal(dn: pl.DataFrame, do: pl.DataFrame) -> bool:
    if dn.columns != do.columns or dn.height != do.height:
        return False
    return all(col_equal(dn[c].to_numpy(), do[c].to_numpy()) for c in dn.columns)


p = _DatasetPaths.resolve(None)
lk = pl.read_parquet(
    Path(p.root) / "processed" / lookup_filename("daily", full_history=True),
    columns=["user_id", "date"],
)
elig = set(zip(lk["user_id"].cast(pl.Utf8).to_list(),
               lk["date"].cast(pl.Utf8).str.slice(0, 10).to_list()))
print(f"eligible_keys: {len(elig)}")

shards = sorted(glob.glob(str(p.daily_hf) + "/data-*.arrow"))[:N_SHARDS]
print(f"sample shards: {len(shards)}")

# ── timeseries: real builder, old vs new, on a temp dir of these shards ──
tmp = Path(tempfile.mkdtemp())
for s in shards:
    (tmp / Path(s).name).symlink_to(s)
ck_new, ck_old = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
build_user_features_chunked(arrow_dir=tmp, checkpoint_dir=ck_new, eligible_keys=elig)
build_user_features_chunked(arrow_dir=tmp, checkpoint_dir=ck_old,
                            max_nonwear_minutes=MAX_NONWEAR, variance_filter=True)

ts_ok = True
ts_new = ts_old = 0
for f in sorted(glob.glob(str(ck_new) + "/*.parquet")):
    dn = pl.read_parquet(f).sort(["user_id", "date"])
    do = pl.read_parquet(str(ck_old / Path(f).name)).sort(["user_id", "date"])
    ts_new += dn.height
    ts_old += do.height
    if not frames_equal(dn, do):
        ts_ok = False
        print(f"  TS DIFF {Path(f).name}: new {dn.shape} vs old {do.shape}")
print(f"timeseries checkpoints: new_rows={ts_new} old_rows={ts_old} byte-identical={ts_ok}")

# ── curve_analysis _accumulate: old vs new on the same shards ──
sn, cn, so_, co = {}, {}, {}, {}
for s in shards:
    _accumulate_arrow_file(Path(s), sn, cn, eligible_keys=elig)
    _accumulate_arrow_file(Path(s), so_, co, max_nonwear_minutes=MAX_NONWEAR, variance_filter=True)
curve_ok = set(sn) == set(so_)
for u in sn:
    for ch in sn[u]:
        if not (np.array_equal(sn[u][ch], so_[u][ch], equal_nan=True)
                and np.array_equal(cn[u][ch], co[u][ch], equal_nan=True)):
            curve_ok = False
print(f"curve accumulate: new_users={len(sn)} old_users={len(so_)} identical={curve_ok}")

shutil.rmtree(tmp)
shutil.rmtree(ck_new)
shutil.rmtree(ck_old)
print(f"\nGATE {'PASSED' if ts_ok and curve_ok else 'FAILED'}: xgboost provider-eligibility "
      f"{'== old wear/variance filter (features byte-identical)' if ts_ok and curve_ok else 'DIFFERS — investigate'}")
