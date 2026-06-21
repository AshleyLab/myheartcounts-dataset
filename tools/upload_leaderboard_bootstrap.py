r"""Upload the bootstrap reference artifact to the OpenMHC leaderboard dataset.

Companion to ``upload_leaderboard_substrate.py``. Where that tool uploads
per-method substrate parquets (one file per method, under
``<track>/<method>.parquet``), this one uploads the **bootstrap reference**
for a track — the Phase-1 long-format per-(method, scenario, channel, draw)
E / R / rank frame that drives skill / rank / fairness CIs:

  * ``bootstrap_draws.parquet`` + ``.meta.json``

Layout in the HF dataset:

    <track>/bootstrap/draws.parquet
    <track>/bootstrap/draws.meta.json

(The ``bootstrap/`` subdir keeps these grouped and clearly separate from the
sibling ``<track>/<method>.parquet`` substrate files. The ``bootstrap_``
prefix is dropped from the destination names since the subdir already
implies it.)

Note: the pooled ``per_user_errors.parquet`` (the BCa LOO substrate) is
deliberately **not** uploaded — it is the concatenation of the per-method
``<track>/<method>.parquet`` files already on HF (``2,376,160 rows =
148,510 rows/method × 16 methods``), so any consumer that needs it can
reconstitute it with one ``pd.concat`` over the per-method files. The
same provenance (seed, n_boot, method list, git commit) lives in
``draws.meta.json`` already.

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


# (source filename in --dir, destination filename in repo, required)
# The parquet file is required; the .meta.json sidecar is uploaded if
# present and skipped (with a warning) if absent.
_FILES: list[tuple[str, str, bool]] = [
    ("bootstrap_draws.parquet", "draws.parquet", True),
    ("bootstrap_draws.parquet.meta.json", "draws.meta.json", False),
]


def main() -> None:
    """Upload bootstrap draw artifacts to the leaderboard dataset."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing bootstrap_draws.parquet (plus its optional "
        ".meta.json sidecar).",
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

    # Documentation files (README.md, SCHEMA.md) co-located with the
    # uploader in the code repo at
    # ``tools/leaderboard_docs/<track>/bootstrap/``. Versioned in git and
    # uploaded alongside the parquet so the HF directory is
    # self-documenting. Missing docs are silently skipped — the parquet
    # upload still works without them.
    docs_dir = Path(__file__).resolve().parent / "leaderboard_docs" / args.track / "bootstrap"
    if docs_dir.is_dir():
        for doc in sorted(docs_dir.glob("*.md")):
            plan.append((doc, f"{args.track}/bootstrap/{doc.name}"))

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
