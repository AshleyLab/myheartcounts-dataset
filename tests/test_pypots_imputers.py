"""Round-trip tests for the PyPOTS-backed imputers.

Each test builds a tiny untrained PyPOTS model, saves it to disk, then loads
it through the public wrapper and runs ``impute()`` on a synthetic batch.
Verifies the protocol contract: output shape and dtype, target positions
imputed (no NaN), non-target positions untouched.

Skipped automatically when pypots is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pypots")


N_CHANNELS = 19
N_STEPS = 48  # small for test speed; arch args below sized to match


def _make_synthetic_batch(
    n: int, n_steps: int = N_STEPS, seed: int = 0, missing_frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Build (data, observed_mask) of shape (n, 19, n_steps), NaN at masked positions."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_CHANNELS, n_steps)).astype(np.float32)
    data[:, 7:, :] = (rng.random((n, 12, n_steps)) < 0.3).astype(np.float32)
    mask = (rng.random((n, N_CHANNELS, n_steps)) > missing_frac).astype(np.float32)
    data = np.where(mask > 0.5, data, np.nan)
    return data, mask


def _make_target_mask(observed_mask: np.ndarray, frac: float = 0.2, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = (observed_mask > 0.5) & (rng.random(observed_mask.shape) < frac)
    return target.astype(np.float32)


def _save_pypots_model(model, tmp_path: Path) -> Path:
    """Save an untrained PyPOTS model so the wrapper has something to load."""
    out = tmp_path / "model.pypots"
    model.save(str(out))
    return out


def _assert_round_trip(out: np.ndarray, data: np.ndarray, target: np.ndarray) -> None:
    """Shape, dtype, no-NaN-at-target, untouched-at-non-target."""
    assert out.shape == data.shape
    assert out.dtype == np.float32
    target_bool = target > 0.5
    assert np.isfinite(out[target_bool]).all()
    # Non-target positions match the input bit-for-bit (including NaN).
    keep = ~target_bool
    np.testing.assert_array_equal(np.isnan(out[keep]), np.isnan(data[keep]))
    finite_keep = keep & np.isfinite(data)
    np.testing.assert_array_equal(out[finite_keep], data[finite_keep])


# ---------------------------------------------------------------------------
# Per-model round-trip tests
# ---------------------------------------------------------------------------


def test_brits_round_trip(tmp_path):
    """BRITSImputer loads a saved BRITS model and imputes target cells while leaving non-targets untouched."""
    from pypots.imputation import BRITS

    from openmhc.imputers import BRITSImputer

    arch = dict(rnn_hidden_size=8)
    _save_pypots_model(
        BRITS(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            **arch,
        ),
        tmp_path,
    )

    imp = BRITSImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        **arch,
    )
    assert imp.name == "pypots_brits"

    data, mask = _make_synthetic_batch(3, seed=0)
    target = _make_target_mask(mask, frac=0.15, seed=1)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_timesnet_round_trip(tmp_path):
    """TimesNetImputer loads a saved TimesNet model and round-trips an impute() call."""
    from pypots.imputation import TimesNet

    from openmhc.imputers import TimesNetImputer

    arch = dict(
        n_layers=1,
        top_k=2,
        d_model=8,
        d_ffn=8,
        n_kernels=2,
        dropout=0.0,
        apply_nonstationary_norm=False,
    )
    _save_pypots_model(
        TimesNet(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            **arch,
        ),
        tmp_path,
    )

    imp = TimesNetImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        **arch,
    )
    assert imp.name == "pypots_timesnet"

    data, mask = _make_synthetic_batch(3, seed=2)
    target = _make_target_mask(mask, frac=0.15, seed=3)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_dlinear_round_trip(tmp_path):
    """DLinearImputer loads a saved DLinear model and round-trips an impute() call."""
    from pypots.imputation import DLinear

    from openmhc.imputers import DLinearImputer

    arch = dict(moving_avg_window_size=5, d_model=8, individual=False)
    _save_pypots_model(
        DLinear(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            **arch,
        ),
        tmp_path,
    )

    imp = DLinearImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        **arch,
    )
    assert imp.name == "pypots_dlinear"

    data, mask = _make_synthetic_batch(3, seed=4)
    target = _make_target_mask(mask, frac=0.15, seed=5)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_fedformer_round_trip(tmp_path):
    """FEDformerImputer round-trips an impute() call, including the version->variant kwarg rename."""
    from pypots.imputation import FEDformer

    from openmhc.imputers import FEDformerImputer

    # Use Wavelets; tiny Fourier configs trip a PyPOTS-internal einsum shape check.
    # PyPOTS's FEDformer calls the basis flavor `version`; our wrapper renames it
    # to `variant` so it doesn't collide with the dataset version.
    shared_arch = dict(
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ffn=16,
        moving_avg_window_size=5,
        dropout=0.0,
        modes=4,
        mode_select="random",
    )
    _save_pypots_model(
        FEDformer(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            version="Wavelets",
            **shared_arch,
        ),
        tmp_path,
    )

    imp = FEDformerImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        variant="Wavelets",
        **shared_arch,
    )
    assert imp.name == "pypots_fedformer"

    data, mask = _make_synthetic_batch(3, seed=6)
    target = _make_target_mask(mask, frac=0.15, seed=7)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


# ---------------------------------------------------------------------------
# Base-class machinery tests
# ---------------------------------------------------------------------------


def test_normalization_round_trip(tmp_path):
    """When stats are provided, predictions are denormalized before copy-back."""
    import json

    from pypots.imputation import BRITS

    from openmhc.imputers import BRITSImputer

    arch = dict(rnn_hidden_size=8)
    _save_pypots_model(
        BRITS(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            **arch,
        ),
        tmp_path,
    )

    stats_path = tmp_path / "stats.json"
    stats = {
        "means": [0.0] * 7 + [0.0] * 12,
        "stds": [2.0] * 7 + [1.0] * 12,
        "channels": list(range(N_CHANNELS)),
        "epsilon": 1e-8,
    }
    # Give continuous channels a non-trivial mean so we can confirm denorm runs.
    stats["means"][:7] = [1.5] * 7
    stats_path.write_text(json.dumps(stats))

    imp = BRITSImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        normalization_stats_path=str(stats_path),
        **arch,
    )

    data, mask = _make_synthetic_batch(2, seed=8)
    target = _make_target_mask(mask, frac=0.1, seed=9)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_directory_path_resolution(tmp_path):
    """Passing a directory containing one .pypots file resolves to that file."""
    from pypots.imputation import BRITS

    from openmhc.imputers import BRITSImputer

    arch = dict(rnn_hidden_size=8)
    _save_pypots_model(
        BRITS(
            n_steps=N_STEPS,
            n_features=N_CHANNELS,
            batch_size=4,
            epochs=1,
            device="cpu",
            **arch,
        ),
        tmp_path,
    )
    # Pass the directory, not the file.
    imp = BRITSImputer(
        model_path=tmp_path,
        version="xs",
        device="cpu",
        inference_batch_size=4,
        n_steps=N_STEPS,
        **arch,
    )
    assert imp._model_file.suffix == ".pypots"


def test_missing_path_raises(tmp_path):
    """A model_path that does not exist raises FileNotFoundError at construction."""
    from openmhc.imputers import BRITSImputer

    with pytest.raises(FileNotFoundError):
        BRITSImputer(
            model_path=tmp_path / "does_not_exist",
            version="xs",
            device="cpu",
            n_steps=N_STEPS,
            rnn_hidden_size=8,
        )


def test_empty_directory_raises(tmp_path):
    """A directory containing no .pypots file raises FileNotFoundError at construction."""
    from openmhc.imputers import BRITSImputer

    with pytest.raises(FileNotFoundError):
        BRITSImputer(
            model_path=tmp_path,
            version="xs",
            device="cpu",
            n_steps=N_STEPS,
            rnn_hidden_size=8,
        )


# ---------------------------------------------------------------------------
# Release manifest (from_release / write_manifest / load_manifest)
# ---------------------------------------------------------------------------


def _build_brits_release(tmp_path, *, with_stats: bool = True, rnn_hidden_size: int = 8):
    """Lay out a tiny BRITS release directory and return its path."""
    import json

    from pypots.imputation import BRITS

    from openmhc.imputers import write_manifest

    m = BRITS(
        n_steps=N_STEPS,
        n_features=N_CHANNELS,
        rnn_hidden_size=rnn_hidden_size,
        batch_size=4,
        epochs=1,
        device="cpu",
    )
    m.save(str(tmp_path / "model.pypots"))

    stats_name: str | None = None
    if with_stats:
        stats_name = "normalization_stats.json"
        (tmp_path / stats_name).write_text(
            json.dumps(
                {
                    "means": [0.0] * N_CHANNELS,
                    "stds": [1.0] * N_CHANNELS,
                    "channels": list(range(7)),
                    "epsilon": 1e-8,
                }
            )
        )

    write_manifest(
        tmp_path,
        kind="brits",
        arch={
            "n_steps": N_STEPS,
            "n_features": N_CHANNELS,
            "rnn_hidden_size": rnn_hidden_size,
        },
        checkpoint="model.pypots",
        normalization_stats=stats_name,
        provenance={"training_run": "test", "dataset_version": "synthetic"},
    )
    return tmp_path


def test_from_release_round_trip(tmp_path):
    """from_release loads a manifest-described BRITS release (recovering arch) and round-trips impute()."""
    from openmhc.imputers import BRITSImputer

    release = _build_brits_release(tmp_path)
    imp = BRITSImputer.from_release(release, version="xs", device="cpu", inference_batch_size=4)
    assert imp.name == "pypots_brits"
    assert imp._rnn_hidden_size == 8

    data, mask = _make_synthetic_batch(2, seed=10)
    target = _make_target_mask(mask, frac=0.1, seed=11)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_from_release_accepts_manifest_file_path(tmp_path):
    """from_release accepts a direct path to the manifest JSON, not just the release directory."""
    from openmhc.imputers import BRITSImputer

    release = _build_brits_release(tmp_path)
    manifest_file = release / "openmhc_manifest.json"
    imp = BRITSImputer.from_release(
        manifest_file, version="xs", device="cpu", inference_batch_size=4
    )
    assert imp.name == "pypots_brits"


def test_from_release_kind_mismatch_raises(tmp_path):
    """Loading a 'brits' release through TimesNetImputer raises ValueError on the kind mismatch."""
    from openmhc.imputers import TimesNetImputer

    release = _build_brits_release(tmp_path)
    with pytest.raises(ValueError, match="kind 'brits'"):
        TimesNetImputer.from_release(release, version="xs", device="cpu")


def test_from_release_optional_stats(tmp_path):
    """Manifests with null normalization_stats produce a working imputer."""
    from openmhc.imputers import BRITSImputer

    release = _build_brits_release(tmp_path, with_stats=False)
    imp = BRITSImputer.from_release(release, version="xs", device="cpu", inference_batch_size=4)
    assert imp._stats is None

    data, mask = _make_synthetic_batch(2, seed=12)
    target = _make_target_mask(mask, frac=0.1, seed=13)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_from_release_runtime_kwargs_override(tmp_path):
    """Runtime kwargs are forwarded to the constructor."""
    from openmhc.imputers import BRITSImputer

    release = _build_brits_release(tmp_path)
    imp = BRITSImputer.from_release(release, version="xs", device="cpu", inference_batch_size=128)
    assert imp._inference_batch_size == 128


def test_release_bundle_is_movable(tmp_path):
    """A release directory loads correctly after being moved (relative paths)."""
    import shutil

    from openmhc.imputers import BRITSImputer

    src = _build_brits_release(tmp_path / "original")
    moved = tmp_path / "elsewhere"
    shutil.copytree(src, moved)
    shutil.rmtree(src)

    imp = BRITSImputer.from_release(moved, version="xs", device="cpu", inference_batch_size=4)
    data, mask = _make_synthetic_batch(2, seed=14)
    target = _make_target_mask(mask, frac=0.1, seed=15)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_load_manifest_rejects_bad_spec_version(tmp_path):
    """load_manifest rejects a manifest with an unsupported spec_version."""
    import json

    from openmhc.imputers import load_manifest

    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps({"spec_version": 999, "kind": "brits", "checkpoint": "x", "arch": {}})
    )
    with pytest.raises(ValueError, match="spec_version"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unknown_kind(tmp_path):
    """load_manifest rejects a manifest whose kind is not a registered imputer."""
    import json

    from openmhc.imputers import load_manifest

    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps({"spec_version": 1, "kind": "saits", "checkpoint": "x", "arch": {}})
    )
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_missing_checkpoint(tmp_path):
    """load_manifest raises FileNotFoundError when the declared checkpoint file is absent."""
    import json

    from openmhc.imputers import load_manifest

    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "kind": "brits",
                "checkpoint": "model.pypots",
                "arch": {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_manifest(tmp_path)


def test_write_then_load_manifest_round_trip(tmp_path):
    """write_manifest output is loadable by load_manifest."""
    from openmhc.imputers import load_manifest, write_manifest

    # Touch a fake checkpoint so the path resolves at load time.
    (tmp_path / "model.pypots").write_bytes(b"\x00")
    write_manifest(
        tmp_path,
        kind="dlinear",
        arch={
            "n_steps": 1440,
            "n_features": 19,
            "moving_avg_window_size": 51,
            "d_model": 256,
        },
        checkpoint="model.pypots",
        normalization_stats=None,
        provenance={"paper_table": "imputation/Table 2"},
    )
    m = load_manifest(tmp_path)
    assert m.kind == "dlinear"
    assert m.arch["d_model"] == 256
    assert m.normalization_stats_path is None
    assert m.provenance["paper_table"] == "imputation/Table 2"
    assert m.checkpoint_path.name == "model.pypots"
