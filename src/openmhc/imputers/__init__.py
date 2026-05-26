"""Reference imputation methods that satisfy the public ``Imputer`` protocol.

Each class fits itself on the official train split in ``__init__`` and
implements ``impute``. They subclass :class:`BaseImputer`, which provides
reusable helpers (channel statistics, metadata access) — but the
``Imputer`` protocol itself is duck-typed and does not require this
base class.

Example:

    >>> import openmhc
    >>> from openmhc.imputers import MeanImputer
    >>> results = openmhc.evaluate_imputation(MeanImputer())

Available reference methods:

- :class:`MeanImputer`, :class:`ModeImputer` — per-channel global statistics.
- :class:`LinearImputer`, :class:`LOCFImputer` — per-sample temporal fills.
- :class:`TemporalMeanImputer`, :class:`TemporalModeImputer` — per-channel,
  per-minute statistics (diurnal patterns).
- :class:`PersonalizedMeanImputer`, :class:`PersonalizedModeImputer`,
  :class:`PersonalizedTemporalMeanImputer` — per-user variants.
- :class:`TorchImputer` — generic wrapper for a pre-trained
  ``torch.nn.Module``.
- :class:`BRITSImputer`, :class:`TimesNetImputer`, :class:`DLinearImputer`,
  :class:`FEDformerImputer` — wrappers around pre-trained PyPOTS
  checkpoints. Requires ``pip install openmhc[pypots]``.
- :class:`LSM2Imputer`, :class:`LSM2WeeklySparseImputer` — wrappers around
  pre-trained LSM2 (Latent Sequence Model v2) Lightning checkpoints.
  Requires ``pip install openmhc[lsm2]``.
- :class:`BaseImputer` — optional base class with shared helpers.
"""

from openmhc.imputers._base import BaseImputer
from openmhc.imputers._personalized_base import PersonalizedImputerBase
from openmhc.imputers._release import Manifest, load_manifest, write_manifest
from openmhc.imputers.linear import LinearImputer
from openmhc.imputers.locf import LOCFImputer
from openmhc.imputers.mean import MeanImputer
from openmhc.imputers.mode import ModeImputer
from openmhc.imputers.personalized import (
    PersonalizedMeanImputer,
    PersonalizedModeImputer,
    PersonalizedTemporalMeanImputer,
)
from openmhc.imputers.lsm2 import LSM2Imputer, LSM2WeeklySparseImputer
from openmhc.imputers.pypots import (
    BRITSImputer,
    DLinearImputer,
    FEDformerImputer,
    TimesNetImputer,
)
from openmhc.imputers.temporal_mean import TemporalMeanImputer
from openmhc.imputers.temporal_mode import TemporalModeImputer
from openmhc.imputers.torch_wrapper import TorchImputer

__all__ = [
    "BaseImputer",
    "PersonalizedImputerBase",
    "MeanImputer",
    "ModeImputer",
    "LinearImputer",
    "LOCFImputer",
    "TemporalMeanImputer",
    "TemporalModeImputer",
    "PersonalizedMeanImputer",
    "PersonalizedModeImputer",
    "PersonalizedTemporalMeanImputer",
    "TorchImputer",
    "BRITSImputer",
    "TimesNetImputer",
    "DLinearImputer",
    "FEDformerImputer",
    "LSM2Imputer",
    "LSM2WeeklySparseImputer",
    "Manifest",
    "load_manifest",
    "write_manifest",
]
