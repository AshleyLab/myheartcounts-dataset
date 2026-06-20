"""LightningModule wrappers around the LSM2 model classes.

Used at inference time via ``load_from_checkpoint`` to restore weights from
the paper's training runs. The training-time methods (``training_step``,
``configure_optimizers``) are retained verbatim from the private repo so
checkpoints load without schema drift.
"""

from openmhc.models.lsm2.modules.module import LSM2Module
from openmhc.models.lsm2.modules.weekly_sparse_module import WeeklySparseLSM2Module

__all__ = ["LSM2Module", "WeeklySparseLSM2Module"]
