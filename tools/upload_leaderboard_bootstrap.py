"""Upload the bootstrap reference artifacts to the OpenMHC leaderboard dataset.

Companion to ``upload_leaderboard_substrate.py``. Where that tool uploads
per-method substrate parquets (one file per method, under
``<track>/<method>.parquet``), this one uploads the **bootstrap reference**
for a track — the artifacts that are derivative-of, not unique-per-, method:

  * ``bootstrap_draws.parquet`` + ``.meta.json`` — the Phase-1 long-format
    per-(method, scenario, channel, draw) E / R / rank frame that drives
    skill / rank / fairness CIs.
  * ``per_user_errors.parquet`` + ``.meta.json`` — the pooled (all-method)
    per-user errors substrate consumed by the BCa LOO jackknife.

Layout in the HF dataset:

    <track>/bootstrap/draws.parquet
    <track>/bootstrap/draws.meta.json
    <track>/bootstrap/per_user_errors.parquet
    <track>/bootstrap/per_user_errors.meta.json

(The ``bootstrap/`` subdir keeps these grouped and clearly separate from the
sibling ``<track>/<method>.parquet`` substrate files. The ``bootstrap_``
prefix is dropped from the destination names since the subdir already
implies it.)

Requires the ``[hf]`` extra (``pip install -e ".[hf]"``) for the
``huggingface_hub`` dependency. Authentication uses the standard
``huggingface_hub`` discovery (``HF_TOKEN`` env or a prior
``huggingface-cli login``).

Usage:
    python tools/upload_leaderboard_bootstrap.py \\
        --dir /scratch/.../openmhc-imputation-eval/paper \\
        --track imputation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"


# (source filename in --dir, destination filename in repo)
# The two parquet files are required; the .meta.json sidecars are uploaded
# if present and skipped (with a warning) if absent.
_FILES: list[tuple[str, str, bool]] = [
    ("bootstrap_draws.parquet", "draws.parquet", True),
    ("bootstrap_draws.parquet.meta.json", "draws.meta.json", False),
    ("per_user_errors.parquet", "per_user_errors.parquet", True),
    ("per_user_errors.parquet.meta.json", "per_user_errors.meta.json", False),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing bootstrap_draws.parquet + per_user_errors.parquet "
        "(plus their optional .meta.json sidecars).",
    )
    p.add_argument(
        "--track",
        default="imputation",
        help="Track subdir in the repo (default: imputation).",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id.")
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo private (default: True).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (paths + sizes) without uploading anything.",
    )
    args = p.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"--dir does not exist or is not a directory: {args.dir}")

    plan: list[tuple[Path, str]] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for src_name, dest_name, required in _FILES:
        src = args.dir / src_name
        if not src.exists():
            (missing_required if required else missing_optional).append(src_name)
            continue
        dest = f"{args.track}/bootstrap/{dest_name}"
        plan.append((src, dest))

    if missing_required:
        raise SystemExit(
            f"Missing required file(s) in {args.dir}: {missing_required}"
        )
    if missing_optional:
        print(f"Note: optional sidecar(s) missing, skipping: {missing_optional}")

    print("Plan:")
    for src, dest in plan:
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  {src.name}  ({size_mb:6.1f} MB)  ->  {args.repo_id}:{dest}")

    if args.dry_run:
        print("--dry-run: no uploads performed.")
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    for src, dest in plan:
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"Uploading {src.name} ({size_mb:.1f} MB)  ->  {dest} ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=(
                f"Add/update {args.track} bootstrap reference: {Path(dest).name}"
            ),
        )
        print(f"  https://huggingface.co/datasets/{args.repo_id}/blob/main/{dest}")


if __name__ == "__main__":
    main()
