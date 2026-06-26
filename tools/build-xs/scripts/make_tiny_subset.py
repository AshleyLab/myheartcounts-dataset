#!/usr/bin/env python3
"""Filter the full OpenMHC dataset down to the XS (593-user) subset.

Reads an *extracted* full tree (mirror of GDrive OpenMHC-Full with the archives
already reconstructed/extracted in place) and writes an OpenMHC-XS tree with the
identical layout but only the users in the XS split.

HF datasets (daily_hf, daily_hourly_hf, hourly_trajectory, minute_trajectory) are
filtered with **pyarrow at the shard level** — the `datasets` library is NOT used,
so this is independent of the datasets version that produced the release. The
original `dataset_info.json`/`state.json` (incl. HF schema metadata) are preserved
so any consumer datasets version can still `load_from_disk` the result.

Filter handlers (ground truth = MHC-benchmark scripts/export/* + verified schemas):
    copy        -> registries / schemas / global stats / docs (verbatim)
    user_first  -> {user_id: ...}            keep keys in XS
    label_first -> {label: {user_id: ...}}   keep inner keys in XS (non-dict labels kept verbatim)
    parquet     -> rows where user_id in XS
    hf          -> pyarrow shard filter on user_id, rewrite as one shard
    hdf5        -> keep <user_id>.h5 files
    split       -> write the XS split json

Every input path is accounted for; unknown files are copied with a loud warning and
flagged in the report. HF outputs are removed before rewrite (idempotent re-runs).
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time, traceback
from pathlib import Path

# Keep any cache off $HOME (15 GB NFS).
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/scratch/users/eggert/openmhc-xs-build/.hf"))
os.environ.setdefault("XDG_CACHE_HOME", os.environ.get("XDG_CACHE_HOME", "/scratch/users/eggert/openmhc-xs-build/.cache"))

# ---- explicit classifications (relative POSIX paths under --full) -------------
HF_DIRS = {
    "data/processed/daily_hf",
    "data/processed/daily_hourly_hf",
    "data/hourly_trajectory",
    "data/minute_trajectory",
}
HDF5_DIR = "data/hdf5"
SPLITS_DIR = "data/splits"
FORECAST_DIR = "data/forecasting_sample_index"

LABEL_FIRST = {
    "data/labels/last_labels.json",
    "data/labels/label_validity.json",
    "data/labels/healthkit_daily.json",
    "data/labels/clip_dates.json",
    "data/labels/context_labels.json",
}
USER_FIRST = {
    "data/labels/enrollment_info.json",
    "data/labels/user_device_info.json",
}
COPY_JSON = {
    "data/labels/label_types.json",
    "data/labels/ordinal_dictionary.json",
    "data/labels/label_pretty_names.json",
    "data/labels/validity_config.json",
    "data/processed/normalization_stats_hourly.json",
    "normalization_stats.json",
    "task_feature_exclusions.json",
}
# user-keyed handlers whose matched-user count of 0 is almost certainly a bug
USER_KEYED = {"user_first", "label_first", "parquet", "hf", "hdf5"}

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_users(path: Path) -> set[str]:
    """Pool user ids across split keys; ignore non-list (meta) values. Identical to 05_verify."""
    d = json.load(open(path))
    if isinstance(d, dict):
        return {u for v in d.values() if isinstance(v, list) for u in v}
    return set(d)

# ---- arrow helpers (version-independent HF filtering) -------------------------
def _read_table(path: Path):
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as m:
        try:
            return pa.ipc.open_stream(m).read_all()
        except Exception:
            pass
    with pa.memory_map(str(path), "r") as m:
        return pa.ipc.open_file(m).read_all()

def _filter_hf_one(src_dir: Path, dst_dir: Path, valset, dry: bool):
    """Filter a single HF save_to_disk dataset dir by user_id; return (rows, n_users)."""
    import pyarrow as pa, pyarrow.compute as pc
    state = json.load(open(src_dir / "state.json"))
    shard_names = [d["filename"] for d in state["_data_files"]]
    assert shard_names, f"{src_dir}: empty _data_files"
    first = _read_table(src_dir / shard_names[0])
    assert "user_id" in first.column_names, f"{src_dir}: no user_id column ({first.column_names})"
    if dry:
        return None, None
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    out_name = "data-00000-of-00001.arrow"
    sink = pa.OSFile(str(dst_dir / out_name), "wb")
    writer = None
    total, seen = 0, set()
    for nm in shard_names:
        t = _read_table(src_dir / nm)
        ft = t.filter(pc.is_in(t.column("user_id"), value_set=valset))
        if ft.num_rows == 0:
            continue
        if writer is None:
            writer = pa.ipc.new_stream(sink, ft.schema)  # ft.schema carries HF metadata
        for b in ft.to_batches():
            writer.write_batch(b)
        total += ft.num_rows
        seen.update(ft.column("user_id").to_pylist())
    if writer is None:                       # nothing matched -> valid empty dataset
        writer = pa.ipc.new_stream(sink, first.schema)
    writer.close(); sink.close()
    # rewrite state.json: single output shard, keep everything else
    state["_data_files"] = [{"filename": out_name}]
    json.dump(state, open(dst_dir / "state.json", "w"))
    # dataset_info.json: preserve features verbatim, fix split counts
    dij = src_dir / "dataset_info.json"
    if dij.exists():
        info = json.load(open(dij))
        for sp in (info.get("splits") or {}).values():
            sp["num_examples"] = total
            sp.pop("num_bytes", None)
        json.dump(info, open(dst_dir / "dataset_info.json", "w"))
    return total, len(seen)

# ---- handlers ----------------------------------------------------------------
def h_copy(src: Path, dst: Path, **_) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return {"handler": "copy"}

def h_user_first(src: Path, dst: Path, users: set, dry: bool, **_) -> dict:
    obj = json.load(open(src))
    assert isinstance(obj, dict), f"{src} not a dict"
    kept = {k: v for k, v in obj.items() if k in users}
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(kept, open(dst, "w"))
    return {"handler": "user_first", "in_users": len(obj), "matched_users": len(kept)}

def h_label_first(src: Path, dst: Path, users: set, dry: bool, **_) -> dict:
    obj = json.load(open(src))
    assert isinstance(obj, dict), f"{src} not a dict"
    kept, nondict, matched = {}, 0, set()
    for lab, inner in obj.items():
        if isinstance(inner, dict):
            f = {u: v for u, v in inner.items() if u in users}
            kept[lab] = f
            matched.update(f)
        else:                       # preserve non-dict label entries verbatim
            kept[lab] = inner
            nondict += 1
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(kept, open(dst, "w"))
    return {"handler": "label_first", "labels": len(kept),
            "nondict_labels": nondict, "matched_users": len(matched)}

def h_parquet(src: Path, dst: Path, users: set, dry: bool, **_) -> dict:
    import pandas as pd
    df = pd.read_parquet(src)
    assert "user_id" in df.columns, f"{src}: no user_id column (cols={list(df.columns)[:8]})"
    out = df[df["user_id"].isin(users)]
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst, index=False)
    return {"handler": "parquet", "in_rows": int(len(df)), "out_rows": int(len(out)),
            "matched_users": int(out["user_id"].nunique())}

def h_hf(src: Path, dst: Path, users: set, dry: bool, **_) -> dict:
    import pyarrow as pa
    valset = pa.array(sorted(users), type=pa.string())
    ddj = src / "dataset_dict.json"
    if ddj.exists():
        splits = json.load(open(ddj)).get("splits") or \
                 [d.name for d in src.iterdir() if (d / "state.json").exists()]
        info = {}
        if not dry:
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            shutil.copy2(ddj, dst / "dataset_dict.json")
        for sp in splits:
            r, u = _filter_hf_one(src / sp, dst / sp, valset, dry)
            info[sp] = {"rows": r, "users": u}
        matched = sum((v["users"] or 0) for v in info.values()) if not dry else None
        return {"handler": "hf", "kind": "DatasetDict", "splits": info, "matched_users": matched}
    r, u = _filter_hf_one(src, dst, valset, dry)
    return {"handler": "hf", "kind": "Dataset", "rows": r, "matched_users": u}

def h_hdf5(src: Path, dst: Path, users: set, dry: bool, **_) -> dict:
    found, missing = 0, 0
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
    for u in sorted(users):
        f = src / f"{u}.h5"
        if f.exists():
            found += 1
            if not dry:
                shutil.copy2(f, dst / f"{u}.h5")
        else:
            missing += 1
    return {"handler": "hdf5", "matched_users": found, "missing": missing}

# ---- json shape auto-detection (fallback for genuinely unknown json) ---------
def detect_json_kind(src: Path, full_users: set, sample: int = 200) -> str:
    try:
        obj = json.load(open(src))
    except Exception:
        return "copy"
    if not isinstance(obj, dict):
        return "copy"
    keys = list(obj.keys())
    if sum(1 for k in keys[:sample] if k in full_users) >= max(1, int(0.3 * min(sample, len(keys)))):
        return "user_first"
    for k in keys[:5]:
        v = obj.get(k)
        if isinstance(v, dict) and sum(1 for ik in list(v.keys())[:sample] if ik in full_users) >= 1:
            return "label_first"
    return "copy"

HANDLERS = {"copy": h_copy, "user_first": h_user_first, "label_first": h_label_first,
            "parquet": h_parquet, "hf": h_hf, "hdf5": h_hdf5}

def classify(rel: str) -> str:
    if rel in HF_DIRS:        return "hf"
    if rel == HDF5_DIR:       return "hdf5"
    if rel in LABEL_FIRST:    return "label_first"
    if rel in USER_FIRST:     return "user_first"
    if rel in COPY_JSON:      return "copy"
    if rel.startswith(FORECAST_DIR + "/") and rel.endswith(".json"):
        return "user_first"   # SPEC: all forecasting_sample_index files are {user_id: [...]}
    if rel.endswith(".parquet"):
        return "parquet"
    if rel.endswith(".md"):
        return "copy"
    if rel.endswith(".json"):
        return "detect"
    return "copy"

def make_readme(stage: Path, users: set) -> None:
    n = len(users)
    (stage / "README.md").write_text(f"""# OpenMHC-XS

A 5% development subset of the OpenMHC dataset — **{n} users** from
`sharable_users_seed42_2026_xs.json` (train 356 / validation 59 / test 178).
Same schema and layout as the full OpenMHC release. Generated by
scripts/make_tiny_subset.py.

## Layout
```
OpenMHC-XS/
├── README.md  normalization_stats.json  task_feature_exclusions.json
├── data/
│   ├── splits/sharable_users_seed42_2026_xs.json
│   ├── labels/            (full label registry, filtered to {n} users)
│   ├── processed/         (lookups + window index, filtered; normalization stats verbatim)
│   └── forecasting_sample_index/
└── archives/
    ├── hdf5_sharable_2026_xs.tar.gz        -> extract to data/hdf5/
    ├── daily_hf_xs.tar.gz                  -> extract to data/processed/
    ├── daily_hourly_hf_xs.tar.gz           -> extract to data/processed/
    ├── hourly_trajectory_xs.tar.gz         -> extract to data/
    └── minute_trajectory_xs.tar.gz         -> extract to data/
```
Normalization statistics are the canonical full-train-split values, shipped
unchanged (do not recompute on the subset).
""")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--full-split", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    full, out = args.full.resolve(), args.out.resolve()
    users = load_users(args.split)
    assert users, "empty XS user set"
    full_users = load_users(args.full_split) if args.full_split and args.full_split.exists() else set()
    tree_full = full / "data/splits/sharable_users_seed42_2026.json"
    if tree_full.exists():
        full_users |= load_users(tree_full)
    if len(full_users) <= len(users):
        full_users = users
        log("WARN: no full-population reference (--full-split / in-tree full split) — "
            "auto-detect of unknown json is disabled-by-degradation; unknown json will be COPIED whole.")
    log(f"XS users: {len(users)} | detect-reference users: {len(full_users)} | dry={args.dry_run}")
    log(f"FULL={full}\nOUT ={out}")

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    report: dict = {"_meta": {"xs_users": len(users), "dry_run": args.dry_run,
                              "full": str(full), "out": str(out)}}

    # worklist: top-level files + known dataset dirs + split + all other data/ files
    worklist: list[str] = [p.name for p in sorted(full.iterdir()) if p.is_file()]
    for d in sorted(HF_DIRS) + [HDF5_DIR]:
        if (full / d).is_dir():
            worklist.append(d)
    if (full / SPLITS_DIR).is_dir():
        worklist.append(SPLITS_DIR + "/__XS_SPLIT__")
    handled = tuple(sorted(HF_DIRS) + [HDF5_DIR, SPLITS_DIR])
    for p in sorted((full / "data").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(full).as_posix()
        if any(rel == h or rel.startswith(h + "/") for h in handled):
            continue
        worklist.append(rel)

    errors, warnings = [], []
    for rel in worklist:
        if rel in skip:
            continue
        if only and not any(rel.startswith(o) for o in only):
            continue
        try:
            if rel.endswith("/__XS_SPLIT__"):
                dst = out / SPLITS_DIR / args.split.name
                if not args.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(args.split, dst)
                report[SPLITS_DIR] = {"handler": "split", "file": args.split.name, "users": len(users)}
                log(f"split        -> {SPLITS_DIR}/{args.split.name} ({len(users)} users)")
                continue
            if rel == "README.md":
                continue   # regenerated at the end
            src = full / rel
            kind = classify(rel)
            if kind == "detect":
                kind = detect_json_kind(src, full_users)
                msg = f"  (auto-detected {rel} -> {kind})"
                if kind == "copy":
                    msg = f"  WARN: unknown json {rel} not recognized as user-keyed -> COPIED WHOLE (review!)"
                    warnings.append(rel)
                log(msg)
            info = HANDLERS[kind](src=src, dst=out / rel, users=users, dry=args.dry_run)
            report[rel] = info
            if kind in USER_KEYED and not args.dry_run and info.get("matched_users") == 0:
                warnings.append(rel)
                log(f"WARN: {rel} matched 0 XS users")
            log(f"{kind:<11} -> {rel}  {info}")
        except Exception as e:
            errors.append(rel)
            report[rel] = {"handler": "ERROR", "error": str(e)}
            log(f"ERROR on {rel}: {e}\n{traceback.format_exc()}")

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        make_readme(out, users)
        report["README.md"] = {"handler": "regen_readme"}

    rep = args.report or (out / "_xs_build_report.json")
    if not args.dry_run:
        rep.parent.mkdir(parents=True, exist_ok=True)
        report["_meta"]["warnings"] = warnings
        report["_meta"]["errors"] = errors
        json.dump(report, open(rep, "w"), indent=2)
    log(f"DONE. artifacts={len(report)-1} errors={len(errors)} warnings={len(warnings)} report={rep}")
    if warnings:
        log(f"WARNINGS (review): {warnings}")
    if errors:
        log(f"FAILED artifacts: {errors}")
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
