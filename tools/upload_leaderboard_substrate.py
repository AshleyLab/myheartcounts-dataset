"""Upload one method's per-user substrate parquet to the OpenMHC leaderboard dataset.

Creates the HF dataset repo if it doesn't exist (private by default), then
uploads the method's substrate to ``<track>/<method>.parquet``. The sidecar
``<track>/<method>.meta.json`` is updated when any of
``--name`` / ``--type`` / ``--submitter`` / ``--subtrack`` /
``--fallback-rate`` is provided, or when ``--results-json`` discovers a
fallback rate to attach (see below). Existing sidecar fields are preserved —
the tool fetches the current sidecar from HF (if any) and merges only the
fields you provide.

Sidecar schema (renderer reads these fields):

| key | type | source |
|---|---|---|
| ``display_name`` | str | ``--name`` |
| ``type`` | str | ``--type`` (``Statistical`` / ``Neural`` / etc.) |
| ``submitter`` | str | ``--submitter`` |
| ``subtrack`` | str | ``--subtrack`` (``single-day`` / ``long-context`` / ...) |
| ``fallback_rate`` | float | ``--fallback-rate`` OR auto-extracted from ``--results-json`` |

The ``fallback_rate`` is the worst-case ``overall_fallback_rate`` across all
``(scenario, split)`` cells in the method's ``results.json`` (mirrors
``openmhc._results.ImputationResults.overall_fallback_rate``). It surfaces
the fraction of target cells the model could not predict and the harness
had to substitute with a channel-aware baseline.

Requires the ``[hf]`` extra (``pip install -e ".[hf]"``) for the
``huggingface_hub`` dependency. Authentication uses the standard
``huggingface_hub`` discovery (``HF_TOKEN`` env or a prior
``huggingface-cli login``).

Usage:
    # Canonical: point at runs/<method>/ and the tool auto-extracts fallback_rate
    python tools/upload_leaderboard_substrate.py \
        --dir paper-verification/per_user \
        --method locf \
        --track imputation \
        --results-json /scratch/.../runs/locf/results.json \
        --name "LOCF (baseline)" --type Statistical --submitter "OpenMHC team"

    # Re-uploading just the fallback rate (other sidecar fields preserved):
    python tools/upload_leaderboard_substrate.py \
        --dir paper-verification/per_user \
        --method locf \
        --track imputation \
        --fallback-rate 0.0
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"


def _worst_case_fallback_rate(results_json: Path) -> float | None:
    """Return ``max(overall_fallback_rate)`` across (scenario, split) in results.json.

    Mirrors :attr:`openmhc._results.ImputationResults.overall_fallback_rate`.
    Returns ``None`` when the field is absent everywhere (legacy runs that
    predate the fallback-tracking feature); ``0.0`` when the field is present
    and every cell reports 0 (modern runs where the harness never had to
    substitute a fallback).
    """
    d = json.loads(results_json.read_text())
    scenarios = d.get("scenarios", d)
    worst = 0.0
    found = False
    for split_map in scenarios.values():
        if not isinstance(split_map, dict):
            continue
        for metrics in split_map.values():
            if not isinstance(metrics, dict):
                continue
            r = metrics.get("overall_fallback_rate")
            if isinstance(r, (int, float)):
                found = True
                if r > worst:
                    worst = float(r)
    return worst if found else None


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


def validate_method_column(parquet_path: Path, method: str) -> None:
    """Fail loudly unless the parquet's ``method`` column is exactly ``method``.

    The leaderboard concatenates every ``imputation/*.parquet`` and groups by the
    ``method`` column, so a substrate whose column disagrees with its upload name
    is mislabeled or collides with another method. The column defaults to
    ``"custom"`` when ``evaluate_imputation`` is run without ``method_name=``;
    this guard turns that silent footgun into an upfront error.
    """
    import pandas as pd

    values = sorted(pd.read_parquet(parquet_path, columns=["method"])["method"].astype(str).unique())
    if values != [method]:
        raise SystemExit(
            f"{parquet_path} has method column {values}, expected ['{method}']. "
            f"Re-run evaluate_imputation(..., method_name='{method}') so the parquet's "
            f"`method` column matches the upload name; the leaderboard groups by that column."
        )


def resolve_fallback_rate(parquet_path: Path, explicit: float | None) -> tuple[float | None, str]:
    """Resolve the ``overall_fallback_rate`` to record in the display sidecar.

    Precedence (issue #39): an explicit ``--fallback-rate`` wins; otherwise read
    it from the substrate's own ``<parquet>.meta.json`` provenance sidecar (which
    ``evaluate_prediction``/``evaluate_*`` write next to the parquet). Returns
    ``(rate, source)``; ``rate`` is ``None`` when neither supplies one, so the
    leaderboard shows "n/a" rather than a fabricated number.
    """
    if explicit is not None:
        return explicit, "--fallback-rate"
    sidecar = Path(f"{parquet_path}.meta.json")
    if sidecar.exists():
        try:
            val = json.loads(sidecar.read_text()).get("overall_fallback_rate")
        except (json.JSONDecodeError, OSError):
            val = None
        if isinstance(val, (int, float)):
            return float(val), sidecar.name
    return None, ""


def main() -> None:
    """Upload one method substrate parquet to the leaderboard dataset."""
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
    p.add_argument(
        "--subtrack",
        default=None,
        help="Sub-track for grouping: 'single-day' or 'long-context' (sidecar).",
    )
    p.add_argument(
        "--fallback-rate",
        type=float,
        default=None,
        help=(
            "Worst-case overall_fallback_rate across (scenario, split) for "
            "this method (sidecar). 0.0 means the harness never had to "
            "substitute a fallback for any predicted cell; >0 means the "
            "model failed to predict that fraction of cells. Mirrors "
            "openmhc._results.ImputationResults.overall_fallback_rate. "
            "Takes precedence over --results-json."
        ),
    )
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help=(
            "Path to the method's results.json (typically "
            "runs/<method>/results.json). When --fallback-rate is not "
            "explicitly set, the worst-case overall_fallback_rate across "
            "all (scenario, split) is auto-extracted from this file and "
            "added to the sidecar. Pass a directory and the tool looks "
            "for results.json inside it."
        ),
    )
    args = p.parse_args()

    src = find_parquet(args.dir, args.method)
    if args.track in ("imputation", "downstream"):
        validate_method_column(src, args.method)
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

    # Auto-extract fallback_rate from results.json if not given explicitly.
    fallback_rate = args.fallback_rate
    if fallback_rate is None and args.results_json is not None:
        rj_path = args.results_json
        if rj_path.is_dir():
            rj_path = rj_path / "results.json"
        if rj_path.exists():
            fallback_rate = _worst_case_fallback_rate(rj_path)
            if fallback_rate is not None:
                print(
                    f"[auto] fallback_rate={fallback_rate:.6f} extracted from {rj_path}"
                )
        else:
            print(f"[auto] --results-json={rj_path} not found; skipping fallback_rate")

    sidecar_provided = {
        "display_name": args.name,
        "type": args.mtype,
        "submitter": args.submitter,
        "subtrack": args.subtrack,
        "fallback_rate": fallback_rate,
    }
    if any(v is not None for v in sidecar_provided.values()):
        # Merge into existing sidecar (if any) so a single-field update like
        # `--fallback-rate` doesn't clobber display_name/type/submitter/subtrack.
        meta_dest = f"{args.track}/{args.method}.meta.json"
        try:
            existing_path = hf_hub_download(
                args.repo_id, meta_dest, repo_type="dataset", force_download=True
            )
            meta = json.loads(Path(existing_path).read_text())
        except (EntryNotFoundError, FileNotFoundError):
            # Brand-new method: build from defaults
            meta = {
                "display_name": args.method,
                "type": "—",
                "submitter": "—",
                "subtrack": "other",
            }
        for key, val in sidecar_provided.items():
            if val is not None:
                meta[key] = val
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
