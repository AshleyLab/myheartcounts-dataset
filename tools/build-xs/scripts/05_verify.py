#!/usr/bin/env python3
"""Independent verification of a staged OpenMHC-XS tree (and optionally the upload tree).

For every artifact in --stage: asserts its user set is a SUBSET of the XS split,
that user-keyed artifacts are NON-EMPTY (0 matched users = build bug), and that the
required inventory is present. HF datasets are read with pyarrow (no datasets dep).
With --upload, also checks the final upload tree has the 5 openable tarballs + small
files. Does NOT trust the build report. Exits non-zero on any hard failure.
"""
from __future__ import annotations
import argparse, json, os, sys, tarfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/scratch/users/eggert/openmhc-xs-build/.hf")

def load_users(p: Path) -> set:
    """Identical pooling to make_tiny_subset.load_users (ignore non-list meta values)."""
    d = json.load(open(p))
    return {u for v in d.values() if isinstance(v, list) for u in v} if isinstance(d, dict) else set(d)

# required small files (fixed names)
EXPECT_FILES = [
    "README.md", "normalization_stats.json", "task_feature_exclusions.json",
    "data/processed/normalization_stats_hourly.json",
    "data/labels/label_types.json", "data/labels/ordinal_dictionary.json",
    "data/labels/label_pretty_names.json", "data/labels/validity_config.json",
    "data/labels/enrollment_info.json", "data/labels/user_device_info.json",
    "data/labels/last_labels.json", "data/labels/label_validity.json",
    "data/labels/healthkit_daily.json", "data/labels/clip_dates.json",
    "data/labels/context_labels.json",
    "data/labels/labels_wide.parquet", "data/labels/label_validity.parquet",
    "data/labels/healthkit_daily.parquet",
    "data/processed/window_index_w7_s7_d5.parquet",
]
# volatile processed lookups: matched by glob (owners rename them, e.g.
# daily_labels_lookup -> daily_labels_lookup_full_history; weekly -> _windowed)
EXPECT_GLOBS = ["data/processed/daily_labels_lookup*.parquet",
                "data/processed/weekly_labels_lookup*.parquet"]
EXPECT_DIRS = ["data/processed/daily_hf", "data/processed/daily_hourly_hf",
               "data/hourly_trajectory", "data/minute_trajectory", "data/hdf5"]
EXPECT_TARBALLS = ["hdf5_sharable_2026_xs.tar.gz", "daily_hf_xs.tar.gz",
                   "daily_hourly_hf_xs.tar.gz", "hourly_trajectory_xs.tar.gz",
                   "minute_trajectory_xs.tar.gz"]
# parquet name-prefixes whose output should never be empty
NONEMPTY_PREFIXES = ("labels_wide", "label_validity", "daily_labels_lookup",
                     "weekly_labels_lookup", "window_index")

def _read_table(path: Path):
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as m:
        try:
            return pa.ipc.open_stream(m).read_all()
        except Exception:
            pass
    with pa.memory_map(str(path), "r") as m:
        return pa.ipc.open_file(m).read_all()

def users_of_parquet(p: Path) -> set:
    import pandas as pd
    return set(pd.read_parquet(p, columns=["user_id"])["user_id"].unique())

def users_of_user_first(p: Path) -> set:
    return set(json.load(open(p)).keys())

def users_of_label_first(p: Path) -> set:
    obj = json.load(open(p))
    return {u for inner in obj.values() if isinstance(inner, dict) for u in inner}

def users_of_hf(p: Path) -> set:
    ddj = p / "dataset_dict.json"
    dirs = [p]
    if ddj.exists():
        sp = json.load(open(ddj)).get("splits") or [d.name for d in p.iterdir() if (d / "state.json").exists()]
        dirs = [p / s for s in sp]
    out = set()
    for d in dirs:
        st = json.load(open(d / "state.json"))
        for df in st["_data_files"]:
            t = _read_table(d / df["filename"])
            if "user_id" in t.column_names:
                out.update(t.column("user_id").to_pylist())
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--upload", type=Path, default=None, help="also verify the final upload tree")
    ap.add_argument("--check-hf", action="store_true")
    args = ap.parse_args()
    xs = load_users(args.split)
    stage = args.stage.resolve()
    fails, summary = [], {}

    def check(name, u, nonempty):
        extra = u - xs
        summary[name] = {"users": len(u), "extra": len(extra)}
        if extra:
            fails.append(f"{name}: {len(extra)} users NOT in XS")
        if nonempty and len(u) == 0:
            fails.append(f"{name}: 0 matched users (empty artifact)")

    # 1) inventory
    missing = [f for f in EXPECT_FILES if not (stage / f).exists()]
    missing += [d + "/" for d in EXPECT_DIRS if not (stage / d).is_dir()]
    for g in EXPECT_GLOBS:
        if not list(stage.glob(g)):
            fails.append(f"missing {g}")
    fdir = stage / "data/forecasting_sample_index"
    nfc = len(list(fdir.glob("*.json"))) if fdir.is_dir() else 0
    if nfc == 0:
        fails.append("forecasting_sample_index/ missing or empty")
    elif nfc != 9:
        summary["_forecast_count"] = f"{nfc} (expected 9)"
    if not (stage / "data/splits" / args.split.name).exists():
        fails.append(f"missing split data/splits/{args.split.name}")
    if missing:
        fails.append(f"missing inventory: {missing}")

    # 2) per-artifact subset + non-empty checks
    for f in sorted(stage.rglob("*.parquet")):
        try:
            ne = any(f.name.startswith(p) for p in NONEMPTY_PREFIXES)
            check(f.relative_to(stage).as_posix(), users_of_parquet(f), nonempty=ne)
        except Exception as e:
            fails.append(f"parquet {f.name}: {e}")
    for name, fn, nonempty in [("enrollment_info", users_of_user_first, True),
                               ("user_device_info", users_of_user_first, True),
                               ("last_labels", users_of_label_first, True),
                               ("label_validity", users_of_label_first, True),
                               ("healthkit_daily", users_of_label_first, False),
                               ("clip_dates", users_of_label_first, False),
                               ("context_labels", users_of_label_first, False)]:
        p = stage / "data/labels" / f"{name}.json"
        if p.exists():
            try:
                check(f"labels/{name}.json", fn(p), nonempty)
            except Exception as e:
                fails.append(f"json {name}: {e}")
    if fdir.is_dir():
        for p in sorted(fdir.glob("*.json")):
            check(f"forecast/{p.name}", set(json.load(open(p)).keys()), nonempty=False)
    hdir = stage / "data/hdf5"
    if hdir.is_dir():
        check("hdf5", {f.stem for f in hdir.glob("*.h5")}, nonempty=True)

    # 3) HF datasets (pyarrow; slow-ish)
    if args.check_hf:
        for d in ["data/processed/daily_hf", "data/processed/daily_hourly_hf",
                  "data/hourly_trajectory", "data/minute_trajectory"]:
            p = stage / d
            if not p.is_dir():
                continue
            try:
                check(d, users_of_hf(p), nonempty=(d.endswith("daily_hf") or d.endswith("trajectory")))
            except Exception as e:
                fails.append(f"hf {d}: {e}")

    # 4) optional: final upload tree
    if args.upload:
        up = args.upload.resolve()
        for t in EXPECT_TARBALLS:
            tp = up / "archives" / t
            if not tp.exists():
                fails.append(f"upload: missing archives/{t}"); continue
            try:
                with tarfile.open(tp, "r:gz") as tf:
                    if not tf.next():
                        fails.append(f"upload: empty tarball {t}")
            except Exception as e:
                fails.append(f"upload: unreadable {t}: {e}")
        for f in ["README.md", "normalization_stats.json", "task_feature_exclusions.json",
                  "data/labels/labels_wide.parquet", "data/splits/" + args.split.name]:
            if not (up / f).exists():
                fails.append(f"upload: missing {f}")

    print(json.dumps(summary, indent=2))
    print(f"\nXS size = {len(xs)}")
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("OK: all artifacts are non-empty subsets of the XS split; inventory complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
