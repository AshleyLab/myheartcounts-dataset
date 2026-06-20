#!/usr/bin/env python
"""Publish staged release bundles to the MyHeartCounts org on Hugging Face Hub.

Each release directory under ``--staging-dir`` is uploaded to a same-named
repo under ``MyHeartCounts/``. The directory layout produced by
``tools/build_manifest.py`` + the staging step in the OpenMHC docs is what
this script expects::

    <staging-dir>/openmhc-brits-imp/
    ├── BRITS.pypots
    ├── normalization_stats.json
    ├── openmhc_manifest.json
    └── README.md

Requires the ``[hf]`` extra (``huggingface_hub``) and a token with write
access to ``MyHeartCounts/*`` repos. Pick one of:

- ``huggingface-cli login`` (token cached in ``~/.cache/huggingface/token``)
- ``export HF_TOKEN=hf_...``

Usage::

    python tools/publish_to_hf.py --staging-dir /path/to/releases-hf
    python tools/publish_to_hf.py --staging-dir /path/to/releases-hf --tag v1.0
    python tools/publish_to_hf.py --staging-dir /path/to/releases-hf \\
        --only openmhc-brits-imp --tag v1.0

The ``--tag`` step creates an immutable git tag on the upload commit so
end users can pin a paper-faithful revision via
``from_release("hf://MyHeartCounts/openmhc-brits-imp@v1.0")``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ORG = "MyHeartCounts"
REQUIRED_FILES = ("openmhc_manifest.json",)

logger = logging.getLogger("publish_to_hf")


def _validate_bundle(bundle_dir: Path) -> None:
    """Check that a bundle has at least the manifest before we touch HF."""
    for required in REQUIRED_FILES:
        if not (bundle_dir / required).exists():
            raise FileNotFoundError(
                f"Bundle {bundle_dir} is missing required file {required!r}"
            )


def publish_one(
    bundle_dir: Path,
    *,
    repo_id: str,
    tag: str | None = None,
    private: bool = False,
    overwrite_tag: bool = False,
) -> str:
    """Create the repo (if absent) and upload the bundle contents.

    Returns the URL of the upload commit.
    """
    from huggingface_hub import HfApi, create_repo

    _validate_bundle(bundle_dir)
    api = HfApi()

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Publish {bundle_dir.name}",
    )
    logger.info("Uploaded %s -> %s", bundle_dir.name, commit.commit_url)

    if tag:
        try:
            api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")
            logger.info("Tagged %s as %s", repo_id, tag)
        except Exception as exc:  # noqa: BLE001 - HF raises a specific 409 here
            if not overwrite_tag:
                logger.warning(
                    "Could not create tag %r on %s (likely exists): %s. "
                    "Re-run with --overwrite-tag to replace.",
                    tag, repo_id, exc,
                )
            else:
                api.delete_tag(repo_id=repo_id, tag=tag, repo_type="model")
                api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")
                logger.info("Replaced tag %s on %s", tag, repo_id)

    return commit.commit_url


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Dir holding one staged bundle subdir per release.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to specific bundle dirnames (repeatable). Default: all.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Tag the upload commit (e.g. v1.0) so users can pin via hf://...@<tag>.",
    )
    p.add_argument("--overwrite-tag", action="store_true")
    p.add_argument("--private", action="store_true", help="Create repos as private.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    bundles = sorted(p for p in args.staging_dir.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        bundles = [b for b in bundles if b.name in wanted]
        missing = wanted - {b.name for b in bundles}
        if missing:
            logger.error("Requested bundles not found: %s", sorted(missing))
            return 2

    if not bundles:
        logger.error("No bundles to publish under %s", args.staging_dir)
        return 2

    for bundle in bundles:
        repo_id = f"{ORG}/{bundle.name}"
        if args.dry_run:
            logger.info("[dry-run] would publish %s -> %s", bundle, repo_id)
            continue
        publish_one(
            bundle,
            repo_id=repo_id,
            tag=args.tag,
            private=args.private,
            overwrite_tag=args.overwrite_tag,
        )

    logger.info("Done. %d bundle(s) processed.", len(bundles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
