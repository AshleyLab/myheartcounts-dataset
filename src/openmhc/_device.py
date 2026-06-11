"""Single source of truth for resolving a torch device specifier.

Used by every public-API class that takes a ``device=`` argument so the
default of ``"auto"`` does the right thing on both GPU and CPU-only hosts
without forcing CUDA at import time.
"""

from __future__ import annotations


def resolve_device(spec: str | None) -> str:
    """Resolve a user-supplied device spec to a concrete torch device string.

    Args:
        spec: One of ``None``, ``"auto"``, ``"cuda"``, ``"cuda:N"``,
            ``"cpu"``, ``"mps"``. ``None`` and ``"auto"`` are equivalent
            and pick CUDA if available, then MPS if available, otherwise
            CPU. Explicit values are returned verbatim (after stripping
            whitespace).

    Returns:
        A concrete device string suitable for ``torch.device(...)``.

    Note:
        ``torch`` is imported lazily so importing this module does not
        pull in the heavy dependency.
    """
    if spec is None:
        spec = "auto"
    spec = spec.strip()
    if spec.lower() != "auto":
        return spec
    import torch  # local import — torch is a heavy dep

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
