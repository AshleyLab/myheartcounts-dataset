"""Tests for the shared device-spec resolver.

Public-API constructors default to ``device="auto"`` so the package works
on CPU-only hosts (no CUDA crash at import or instantiation time). The
resolver also accepts ``None`` as a synonym for ``"auto"`` and passes
explicit values through unchanged.
"""

from __future__ import annotations

import pytest

from openmhc._device import resolve_device


def test_resolve_explicit_cpu_is_passthrough():
    """An explicit ``"cpu"`` spec is returned unchanged."""
    assert resolve_device("cpu") == "cpu"


def test_resolve_explicit_cuda_is_passthrough():
    """Explicit CUDA specs pass through verbatim regardless of CUDA availability."""
    # Explicit values are honored verbatim regardless of CUDA availability —
    # the caller has explicitly opted in.
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_strips_whitespace():
    """Surrounding whitespace is stripped from the device spec."""
    assert resolve_device("  cpu  ") == "cpu"


def test_resolve_auto_returns_concrete_device():
    """``"auto"`` and ``None`` resolve to a concrete device.

    The result must be one ``torch.device(...)`` accepts.
    """
    import torch

    for spec in (None, "auto", "AUTO"):
        out = resolve_device(spec)
        assert out in {"cuda", "mps", "cpu"}, f"unexpected resolved device: {out!r}"
        # Round-trip through torch.device to guarantee it's valid.
        torch.device(out)


@pytest.mark.skipif(
    __import__("torch").cuda.is_available(),
    reason="Test that auto picks CPU; meaningless on a CUDA host",
)
def test_resolve_auto_picks_cpu_when_no_gpu():
    """On a CPU-only host, ``auto`` produces ``cpu`` or ``mps``, never ``cuda``."""
    assert resolve_device("auto") in {"cpu", "mps"}
