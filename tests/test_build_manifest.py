"""Tests for tools/build_manifest.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import build_manifest  # noqa: E402


def _write_brits_config(repo_root: Path, ckpt_relpath: str) -> Path:
    config = {
        "seed": 42,
        "method": {
            "type": "pypots",
            "pypots": {
                "model_path": ckpt_relpath,
                "model_name": "brits",
                "device": "cuda",
                "inference_batch_size": 400,
                "normalization_stats_path": "data/processed/normalization_stats.json",
                "n_steps": 1440,
                "n_features": 19,
                "rnn_hidden_size": 128,
                # Fields from PyPOTSMethodConfig that don't apply to BRITS;
                # the converter should drop them.
                "n_layers": 2,
                "top_k": 5,
                "d_model": 64,
            },
        },
        "output": {"experiment_name": "pypots_brits"},
    }
    eval_run = repo_root / "results/imputation_eval/brits_max91d_20260415_010915"
    eval_run.mkdir(parents=True)
    cfg = eval_run / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    return cfg


def _seed_private_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake private-repo layout with a (dummy) checkpoint + stats."""
    repo = tmp_path / "MHC-benchmark"
    ckpt_dir = repo / "models/pypots/brits/20260409_T005730"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "BRITS_epoch5_MAE0.0945.pypots"
    ckpt.write_bytes(b"\x00fake-checkpoint")  # contents irrelevant for the test

    stats = repo / "data/processed/normalization_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "means": [0.0] * 19,
                "stds": [1.0] * 19,
                "channels": list(range(7)),
                "epsilon": 1e-8,
            }
        )
    )
    return repo, ckpt


def test_build_release_writes_manifest_and_copies_files(tmp_path):
    repo, ckpt = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(
        repo, "models/pypots/brits/20260409_T005730/BRITS_epoch5_MAE0.0945.pypots"
    )
    out_dir = tmp_path / "releases"

    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=out_dir, release_name="brits-test"
    )
    assert result.status == "built"
    assert result.release_dir == out_dir / "brits-test"

    release = result.release_dir
    assert (release / "model.pypots").read_bytes() == ckpt.read_bytes()
    assert (release / "normalization_stats.json").exists()

    manifest = json.loads((release / "openmhc_manifest.json").read_text())
    assert manifest["kind"] == "brits"
    assert manifest["arch"] == {
        "n_steps": 1440,
        "n_features": 19,
        "rnn_hidden_size": 128,
    }
    # Unrelated fields are dropped.
    assert "n_layers" not in manifest["arch"]
    assert manifest["checkpoint"] == "model.pypots"
    assert manifest["normalization_stats"] == "normalization_stats.json"
    assert manifest["provenance"]["training_run"] == "20260409_T005730"
    assert manifest["provenance"]["eval_run"] == "brits_max91d_20260415_010915"
    assert manifest["provenance"]["experiment_name"] == "pypots_brits"


def test_build_release_resolves_directory_checkpoint(tmp_path):
    """A model_path pointing at a directory picks the first .pypots inside."""
    repo, _ = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(repo, "models/pypots/brits/20260409_T005730")
    out_dir = tmp_path / "releases"
    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=out_dir, release_name="brits-test"
    )
    assert result.status == "built"
    assert (result.release_dir / "model.pypots").exists()


def test_build_release_skips_wandb_refs(tmp_path):
    repo, _ = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(repo, "wandb:MHC_Dataset/mhc-pypots-dlinear/dlinear:v48")
    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=tmp_path / "releases", release_name="x"
    )
    assert result.status == "skipped"
    assert "wandb" in result.reason.lower()


def test_build_release_skips_unsupported_kind(tmp_path):
    repo, _ = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(
        repo, "models/pypots/brits/20260409_T005730/BRITS_epoch5_MAE0.0945.pypots"
    )
    # Mutate the config to use an unsupported kind.
    config = yaml.safe_load(cfg.read_text())
    config["method"]["pypots"]["model_name"] = "saits"
    cfg.write_text(yaml.safe_dump(config))

    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=tmp_path / "releases", release_name="x"
    )
    assert result.status == "skipped"
    assert "saits" in result.reason


def test_build_release_refuses_overwrite_without_flag(tmp_path):
    repo, _ = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(
        repo, "models/pypots/brits/20260409_T005730/BRITS_epoch5_MAE0.0945.pypots"
    )
    out_dir = tmp_path / "releases"
    build_manifest.build_release(
        cfg, source_repo=repo, output_dir=out_dir, release_name="brits-test"
    )
    again = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=out_dir, release_name="brits-test"
    )
    assert again.status == "skipped"
    assert "already exists" in again.reason
    fresh = build_manifest.build_release(
        cfg,
        source_repo=repo,
        output_dir=out_dir,
        release_name="brits-test",
        overwrite=True,
    )
    assert fresh.status == "built"


def test_manifest_round_trips_through_load_manifest(tmp_path):
    """An emitted manifest is loadable via openmhc.imputers.load_manifest."""
    from openmhc.imputers import load_manifest

    repo, _ = _seed_private_repo(tmp_path)
    cfg = _write_brits_config(
        repo, "models/pypots/brits/20260409_T005730/BRITS_epoch5_MAE0.0945.pypots"
    )
    result = build_manifest.build_release(
        cfg,
        source_repo=repo,
        output_dir=tmp_path / "releases",
        release_name="brits-test",
    )
    m = load_manifest(result.release_dir)
    assert m.kind == "brits"
    assert m.arch["rnn_hidden_size"] == 128
    assert m.checkpoint_path.name == "model.pypots"
    assert m.normalization_stats_path is not None


# ---------------------------------------------------------------------------
# LSM2 cases (Lightning .ckpt input)
# ---------------------------------------------------------------------------


def _seed_lsm2_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake private-repo layout with a Lightning .ckpt + eval config."""
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    import torch

    repo = tmp_path / "MHC-benchmark"
    ckpt_dir = repo / "models/mae/20260409_T005730"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "lsm2-paper.ckpt"

    # Minimal Lightning-style checkpoint: state_dict (empty), hparams, embedded
    # normalization stats. The converter never instantiates the model — it
    # just reads hparams and the LightningDataModule block.
    torch.save(
        {
            "state_dict": {},
            "hyper_parameters": {
                "seq_length": 1440,
                "patch_size": 10,
                "in_channels": 19,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "decoder_embed_dim": 256,
                "decoder_depth": 4,
                "decoder_num_heads": 4,
                "mlp_ratio": 4.0,
                "mask_ratio": 0.5,
            },
            "pytorch-lightning_version": "2.0.0",
            "LightningDataModule": {
                "normalization_stats": {
                    "mean_prior": [1.5] * 7 + [0.0] * 12,
                    "std_prior": [2.5] * 7 + [1.0] * 12,
                    "prior_count": 1e12,
                }
            },
        },
        ckpt_path,
    )

    cfg = {
        "method": {
            "type": "mae",  # private-repo value; converter normalizes to "lsm2"
            "lsm2": {
                "checkpoint_path": str(ckpt_path.relative_to(repo)),
                "device": "cuda",
                "inference_batch_size": 128,
                "normalization_prior_count": 1.0e12,
            },
        },
        "output": {"experiment_name": "pypots_lsm2_daily"},
    }
    eval_run = repo / "results/imputation_eval/mae_daily_nodropout_max91d_20260413_222844"
    eval_run.mkdir(parents=True)
    cfg_path = eval_run / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return repo, cfg_path


def test_build_release_lsm2_extracts_arch_and_stats(tmp_path):
    """LSM2 build: arch read from ckpt hparams, stats extracted to sibling JSON."""
    repo, cfg = _seed_lsm2_repo(tmp_path)
    out_dir = tmp_path / "releases"

    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=out_dir, release_name="lsm2-daily"
    )
    assert result.status == "built", result.reason

    release = result.release_dir
    # Checkpoint copied as model.ckpt (not model.pypots).
    assert (release / "model.ckpt").exists()
    # Stats extracted from the .ckpt's LightningDataModule block.
    stats_file = release / "normalization_stats.json"
    assert stats_file.exists()
    stats = json.loads(stats_file.read_text())
    assert stats["means"][:7] == [1.5] * 7
    assert stats["stds"][:7] == [2.5] * 7

    manifest = json.loads((release / "openmhc_manifest.json").read_text())
    assert manifest["kind"] == "lsm2"  # mae → lsm2 alias applied
    assert manifest["arch"]["embed_dim"] == 384
    assert manifest["arch"]["depth"] == 12
    assert manifest["checkpoint"] == "model.ckpt"
    assert manifest["normalization_stats"] == "normalization_stats.json"


def test_build_release_lsm2_round_trips_through_load_manifest(tmp_path):
    from openmhc.imputers import load_manifest

    repo, cfg = _seed_lsm2_repo(tmp_path)
    result = build_manifest.build_release(
        cfg, source_repo=repo, output_dir=tmp_path / "releases", release_name="lsm2-daily"
    )
    m = load_manifest(result.release_dir)
    assert m.kind == "lsm2"
    assert m.arch["embed_dim"] == 384
    assert m.checkpoint_path.name == "model.ckpt"
