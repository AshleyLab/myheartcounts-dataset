"""Reference Track-1 (outcome-prediction) encoders that load released bundles.

Each class loads a published checkpoint bundle (local directory or ``hf://``
URI) via :meth:`from_release` and satisfies the public :class:`~openmhc.Method`
contract, delegating to the internal downstream-evaluation engine.

Example:
    >>> import openmhc
    >>> from openmhc.encoders import WBM
    >>> enc = WBM.from_release("hf://MyHeartCounts/openmhc-wbm-dp")
    >>> results = openmhc.evaluate_prediction(enc, version="full")

Available reference models:

- :class:`WBM` — the reported WBM model (Mamba2 contrastive SSL encoder +
  Linear fallback). Running the encoder requires the CUDA-only ``mamba_ssm``
  kernels.

Released bundles live under ``MyHeartCounts/openmhc-<model>-dp`` on the Hugging
Face Hub.
"""

from openmhc.encoders._release import Manifest, load_manifest, write_manifest
from openmhc.encoders.wbm import WBM

__all__ = [
    "WBM",
    "Manifest",
    "load_manifest",
    "write_manifest",
]
