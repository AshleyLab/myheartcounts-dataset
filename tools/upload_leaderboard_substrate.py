"""Upload one method's per-user substrate parquet to the OpenMHC leaderboard dataset.

Creates the HF dataset repo if it doesn't exist (private by default), then
uploads the method's substrate to ``<track>/<method>.parquet``. When any of
``--name`` / ``--type`` / ``--submitter`` is given, also writes a
``<track>/<method>.meta.json`` display sidecar (name, type, submitter) that the
leaderboard reads to render the row.

Requires the ``[hf]`` extra (``pip install -e ".[hf]"``) for the
``huggingface_hub`` dependency. Authentication uses the standard
``huggingface_hub`` discovery (``HF_TOKEN`` env or a prior
``huggingface-cli login``).

Usage:
    python tools/upload_leaderboard_substrate.py \
        --dir src/openmhc/data/baselines \
        --method locf \
        --track imputation \
        --name "LOCF (baseline)" --type Statistical --submitter "OpenMHC team"
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"


def find_parquet(dir_path: Path, method: str) -> Path:
    """Locate the substrate parquet for ``method`` inside ``dir_path``.

    Prefers an exact stem match (e.g. ``method='mean'`` → ``mean.parquet``);
    falls back to a substring match on the filename; finally falls back to the
    sole parquet in the directory. Errors if the choice is ambiguous.

    The exact-stem step matters when method names share prefixes (e.g. ``mean``
    is a substring of ``temporal_mean`` and ``personalized_mean``); a pure
    substring match would mis-flag the lookup as ambiguous.
    """
    parquets = sorted(dir_path.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No .parquet files in {dir_path}")
    exact = [p for p in parquets if p.stem == method]
    if len(exact) == 1:
        return exact[0]
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
    p.add_argument("--name", default=None, help="Display name (writes a <method>.meta.json sidecar).")
    p.add_argument("--type", dest="mtype", default=None, help="Method type, e.g. 'Deep Learning' (sidecar).")
    p.add_argument("--submitter", default=None, help="Submitter / team for attribution (sidecar).")
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

    if args.name or args.mtype or args.submitter:
        meta = {
            "display_name": args.name or args.method,
            "type": args.mtype or "—",
            "submitter": args.submitter or "—",
        }
        meta_dest = f"{args.track}/{args.method}.meta.json"
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(meta, indent=2).encode("utf-8")),
            path_in_repo=meta_dest,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add/update {args.track} metadata: {args.method}",
        )
        print(f"Uploaded sidecar  ->  {args.repo_id}:{meta_dest}")


if __name__ == "__main__":
    main()
