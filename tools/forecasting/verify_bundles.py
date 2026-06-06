#!/usr/bin/env python
"""Smoke-verify staged forecasting release bundles via the public API.

Loads each requested bundle with ``<Wrapper>.from_release`` and runs a single
synthetic ``predict(history, horizon)`` call, checking output shape, finiteness,
and — for neural models — that the train-split scaler was loaded (co-located in
the bundle) and predictions land in raw value space rather than standardized.

Run from any cwd with model deps available, putting the repo src on PYTHONPATH so
the patched code is used, e.g. in the mhc-benchmark env::

    PYTHONPATH=~/myheartcounts-dataset/src /opt/conda/envs/mhc-benchmark/bin/python \
        ~/myheartcounts-dataset/tools/forecasting/verify_bundles.py \
        --staging-dir ~/myheartcounts-dataset/releases-fc --only dlinear segrnn mixlinear chronos2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WRAPPERS = {
    "dlinear": ("neural", "DLinearForecaster"),
    "segrnn": ("neural", "SegRNNForecaster"),
    "mixlinear": ("neural", "MixLinearForecaster"),
    "chronos2": ("foundation", "Chronos2Forecaster"),
    "toto": ("foundation", "TotoForecaster"),
}


def _get_wrapper(name: str):
    import openmhc.forecasters as f

    return getattr(f, WRAPPERS[name][1])


def verify_one(kind: str, bundle: Path, device: str, history_len: int, horizon: int) -> dict:
    """Load one bundle via from_release and run a synthetic predict."""
    wrapper = _get_wrapper(kind)
    fc = wrapper.from_release(str(bundle), device=device)

    rng = np.random.default_rng(0)
    # Raw-scale synthetic history: nonnegative, channel-varying magnitudes.
    history = (rng.random((19, history_len)).astype("float32") * 200.0)
    out = fc.predict(history, horizon)

    res = {
        "kind": kind,
        "out_shape": tuple(out.shape),
        "finite": bool(np.isfinite(out).all()),
        "dtype": str(out.dtype),
        "abs_mean": float(np.abs(out).mean()),
    }
    # Neural models must have loaded the co-located scaler (else outputs are standardized).
    model = getattr(fc, "_model", None)
    scaler = getattr(model, "scaler_stats", None) if model is not None else None
    if WRAPPERS[kind][0] == "neural":
        res["scaler_loaded"] = scaler is not None
    return res


def main(argv: list[str] | None = None) -> int:
    """Verify the requested bundles and print a per-model report."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging-dir", type=Path, default=Path("releases-fc"))
    p.add_argument("--only", nargs="+", default=list(WRAPPERS), choices=list(WRAPPERS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--history-len", type=int, default=336)
    p.add_argument("--horizon", type=int, default=24)
    args = p.parse_args(argv)

    failures = 0
    for kind in args.only:
        bundle = args.staging_dir / f"openmhc-{kind}-fc"
        try:
            res = verify_one(kind, bundle, args.device, args.history_len, args.horizon)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAIL  {kind}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        ok = (
            res["out_shape"] == (19, args.horizon)
            and res["finite"]
            and res.get("scaler_loaded", True)
        )
        flag = "OK  " if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"{flag} {kind}: {res}")

    print(f"\n{'ALL OK' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
