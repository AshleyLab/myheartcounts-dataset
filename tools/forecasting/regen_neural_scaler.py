#!/usr/bin/env python
"""Regenerate the train-split StandardScaler stats for the neural forecasters.

DLinear / SegRNN / MixLinear were trained on standardized ``history_cf`` windows
and need the *training-time* StandardScaler to inverse-transform predictions back
to real units. That scaler is content-addressed by the training data/model config
and the cache that held it has been cleaned up, so we regenerate it deterministically
by replaying the exact training-time fit:

    rows = build_history_cf_rows(train_split, features, model)   # train split only
    scaler = fit_from_history_cf_rows(rows, n_channels=n_features)

All three neural checkpoints share the same scaler (the cache hash ignores model
architecture), so one fit — driven by any of their ``training_config.json`` files —
serves all three bundles.

Run from the private repo root (so the relative dataset paths resolve) with an
environment that has ``forecasting_evaluation`` + the dataset (the ``mhc-benchmark``
conda env)::

    cd ~/MHC-benchmark
    /opt/conda/envs/mhc-benchmark/bin/python \
        ~/myheartcounts-dataset/tools/forecasting/regen_neural_scaler.py \
        --training-config ~/myheartcounts-dataset/forecasting_model_ckpt/dlinear_HPO_scale/training_config.json \
        --output ~/myheartcounts-dataset/.merge_cache/standard_scaler_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main(argv: list[str] | None = None) -> int:
    """Fit and save the train-split StandardScaler for the neural forecasters."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Override dataset num_workers (1 disables multiprocessing in filter/map).",
    )
    args = p.parse_args(argv)

    from forecasting_evaluation.config import DataConfig
    from forecasting_evaluation.data.data_loader import ForecastingDataLoader
    from forecasting_evaluation.data.online_dataset import build_history_cf_rows
    from forecasting_evaluation.data.standard_scaler import (
        fit_from_history_cf_rows,
    )

    cfg = json.loads(Path(args.training_config).expanduser().read_text())
    data_raw = cfg["data"]
    model_raw = cfg["model"]
    features_raw = cfg.get("features", {"channel": "all"})

    # Build a DataConfig with only the fields it declares (ignore extras).
    data_fields = set(DataConfig.__dataclass_fields__)
    data_cfg = DataConfig(**{k: v for k, v in data_raw.items() if k in data_fields})
    if hasattr(data_cfg, "num_workers"):
        data_cfg.num_workers = int(args.num_workers)
    model_cfg = SimpleNamespace(**model_raw)
    features_cfg = SimpleNamespace(**features_raw)
    n_features = int(model_raw["n_features"])

    print(f"n_steps={model_raw['n_steps']} n_pred_steps={model_raw['n_pred_steps']} "
          f"n_features={n_features} sample_index={data_raw['sample_index_file']}")
    print("Loading splits ...")
    loader = ForecastingDataLoader(data_cfg)
    train_split, _val, _test = loader.load_splits()
    print(f"train split size: {len(train_split)} users/rows")

    print("Building train-split history_cf rows ...")
    rows = build_history_cf_rows(
        split_ds=train_split,
        features_config=features_cfg,
        model_config=model_cfg,
    )
    print(f"built {len(rows)} history_cf rows; fitting StandardScaler ...")
    scaler = fit_from_history_cf_rows(rows, n_channels=n_features)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    scaler.save_stats_json(out)
    saved = json.loads(out.read_text())

    means = saved["means"]
    counts = saved.get("valid_counts", [])
    print(f"wrote {out}")
    print(f"  fit_scope={saved.get('fit_scope')} n_channels={saved.get('n_channels')}")
    print(f"  channel-0 mean={means[0]:.4f} std={saved['stds'][0]:.4f} "
          f"valid_count={counts[0] if counts else '?'}")
    print(f"  channel-6 mean={means[6]:.4f} std={saved['stds'][6]:.4f}")
    if counts and counts[0] == 0:
        print("WARNING: channel-0 valid_count is 0 — scaler looks degenerate!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
