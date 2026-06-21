"""Upload one method's per-user substrate parquet to the OpenMHC leaderboard dataset.

Creates the HF dataset repo if it doesn't exist (private by default), then
uploads the method's substrate to ``<track>/<method>.parquet``.

Usage:
    python tools/upload_leaderboard_substrate.py \
        --dir src/openmhc/data/baselines \
        --method locf \
        --track imputation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"


def find_parquet(dir_path: Path, method: str) -> Path:
    """Locate the substrate parquet for ``method`` inside ``dir_path``.

    Prefers a file whose name contains the method; otherwise falls back to the
    sole parquet in the directory. Errors if the choice is ambiguous.
    """
    parquets = sorted(dir_path.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No .parquet files in {dir_path}")
    named = [p for p in parquets if method in p.name]
    if len(named) == 1:
        return named[0]
    if len(parquets) == 1:
        return parquets[0]
    raise SystemExit(
        f"Ambiguous: {len(parquets)} parquet files in {dir_path}, "
        f"{len(named)} matching method '{method}'. Narrow it down."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dir", required=True, type=Path, help="Directory containing the method's substrate parquet."
    )
    p.add_argument("--method", required=True, help="Method name (used as the destination filename).")
    p.add_argument("--track", default="imputation", help="Track subdir in the repo (default: imputation).")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id.")
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo private (default: True).",
    )
    args = p.parse_args()

    src = find_parquet(args.dir, args.method)
    dest = f"{args.track}/{args.method}.parquet"

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=dest,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Add/update {args.track} substrate: {args.method}",
    )
    print(f"Uploaded {src}  ->  {args.repo_id}:{dest}")
    print(f"  https://huggingface.co/datasets/{args.repo_id}/blob/main/{dest}")


if __name__ == "__main__":
    main()
