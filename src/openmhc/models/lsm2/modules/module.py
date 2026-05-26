"""LSM2Module - LightningModule for MAE SSL pretraining on daily data."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import instantiate_class
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch.optim import AdamW

from openmhc.models.lsm2 import LSM2ViT1D, create_inherited_mask

__all__ = ["LSM2Module"]


class LSM2Module(pl.LightningModule):
    """Lightning Module for self-supervised pre-training with MAE-ViT.

    Uses LSM2ViT1D with Adaptive and Inherited Masking (AIM)
    for masked autoencoding on daily wearable sensor data (19 channels × 1440 minutes).

    The model handles both:
    - Artificial masking: For self-supervised learning (random/temporal/sensor masking)
    - Inherited masking: For real-world missing data (detected via NaN values)

    Args:
        seq_length: Length of input sequence (default: 1440 minutes per day).
        patch_size: Size of each patch (default: 10 minutes).
        in_channels: Number of input channels (default: 19 for all channels).
        embed_dim: Embedding dimension for encoder.
        depth: Number of transformer blocks in encoder.
        num_heads: Number of attention heads in encoder.
        decoder_embed_dim: Embedding dimension for decoder.
        decoder_depth: Number of transformer blocks in decoder.
        decoder_num_heads: Number of attention heads in decoder.
        mlp_ratio: MLP hidden dimension ratio.
        norm_pix_loss: Whether to normalize pixels before computing loss.
        dropout_removal_ratio: Fraction of tokens to physically remove.
        mask_ratio: Fraction of tokens to mask artificially.
        use_hybrid_loss: Whether to use hybrid MSE+BCE loss (MSE for continuous,
            BCE for binary channels). Default: True.
        continuous_channels: Tuple of channel indices to use MSE loss for.
            Default: (0, 1, 2, 3, 4, 5, 6) for continuous metrics.
        channel_weights: Optional list of per-channel weights for the loss.
            Must have length equal to in_channels. Default: None (uniform weights).
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for AdamW.
        scheduler: Learning rate scheduler type ("cosine", "exponential", or "constant").
        warmup_ratio: Fraction of steps for warmup.
        lr_scheduler: Optional scheduler config dict (alternative to scheduler param).

    Example:
        >>> model = LSM2Module(learning_rate=1e-4, use_hybrid_loss=True)
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        # Model parameters
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_pix_loss: bool = False,
        dropout_removal_ratio: float = 0.5,
        mask_ratio: float = 0.5,
        # Hybrid loss parameters
        use_hybrid_loss: bool = True,
        continuous_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        channel_weights: list[float] | None = None,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        scheduler: str = "cosine",
        warmup_ratio: float = 0.05,
        eta_min: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.95),
        lr_scheduler: dict | None = None,
        # Per-sample residual storage
        store_residuals: bool = False,
        residuals_dir: str | None = None,
    ):
        """Initialize MAE module.

        Args:
            seq_length: Sequence length.
            patch_size: Patch size.
            in_channels: Input channels.
            embed_dim: Embedding dimension.
            depth: Encoder depth.
            num_heads: Number of attention heads.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Decoder depth.
            decoder_num_heads: Decoder num heads.
            mlp_ratio: MLP ratio.
            qkv_bias: Whether to use bias term in QKV projections.
            norm_pix_loss: Normalize pixel loss.
            dropout_removal_ratio: Dropout removal ratio.
            mask_ratio: Masking ratio.
            use_hybrid_loss: Use hybrid loss.
            continuous_channels: Continuous channel indices.
            channel_weights: Per-channel weights.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            scheduler: Scheduler type.
            warmup_ratio: Warmup ratio.
            eta_min: Minimum eta.
            betas: Adam beta coefficients.
            lr_scheduler: Learning rate scheduler config.
            store_residuals: Store residuals.
            residuals_dir: Residuals directory.
        """
        super().__init__()
        self.save_hyperparameters()

        # Model: LSM2ViT1D
        self.model = LSM2ViT1D(
            seq_length=seq_length,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_pix_loss=norm_pix_loss,
            dropout_removal_ratio=dropout_removal_ratio,
            mask_ratio=mask_ratio,
            use_hybrid_loss=use_hybrid_loss,
            continuous_channels=continuous_channels,
            channel_weights=channel_weights,
        )

        # Store patch_size for inherited mask creation
        self.patch_size = patch_size

        # Per-sample residual storage
        self._store_residuals = store_residuals
        self._residuals_dir = Path(residuals_dir) if residuals_dir else None
        self._train_residuals: list[dict[str, torch.Tensor]] = []
        self._val_residuals: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        x: torch.Tensor,
        inherited_mask: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, T).
            inherited_mask: Optional mask of shape (B, num_patches).
            return_per_sample: If True, also return per-sample losses dict.

        Returns:
            Tuple of (loss, predictions, total_mask) or
            (loss, predictions, total_mask, per_sample_losses) when return_per_sample=True.
        """
        return self.model(x, inherited_mask, return_per_sample=return_per_sample)

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Shared logic for training/validation/test steps.

        Args:
            batch: Dict with "X" of shape (B, C, T).

        Returns:
            Tuple of (loss, total_mask, per_sample_losses | None).
        """
        x = batch["X"]  # (B, C, T) with NaNs for missing data

        # Create inherited mask from NaN values
        inherited_mask = create_inherited_mask(x, self.patch_size)

        # Fill NaNs with 0 (model cannot process NaNs)
        x_filled = torch.nan_to_num(x, nan=0.0)

        # Forward pass (model computes loss internally)
        if self._store_residuals:
            loss, pred, total_mask, per_sample_losses = self(
                x_filled, inherited_mask, return_per_sample=True
            )
            return loss, total_mask, per_sample_losses
        else:
            loss, pred, total_mask = self(x_filled, inherited_mask)
            return loss, total_mask, None

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dict with "X".
            batch_idx: Index of current batch.

        Returns:
            Loss value.
        """
        loss, total_mask, per_sample_losses = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "train/mask_ratio",
            total_mask.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

        # Accumulate per-sample residuals
        if self._store_residuals and per_sample_losses is not None:
            self._train_residuals.append(
                {
                    "idx": batch["idx"].detach().cpu(),
                    "continuous_loss": per_sample_losses["continuous_loss"].detach().cpu(),
                    "binary_loss": per_sample_losses["binary_loss"].detach().cpu(),
                }
            )

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dict with "X".
            batch_idx: Index of current batch.
        """
        loss, total_mask, per_sample_losses = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "val/mask_ratio",
            total_mask.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

        # Accumulate per-sample residuals
        if self._store_residuals and per_sample_losses is not None:
            self._val_residuals.append(
                {
                    "idx": batch["idx"].detach().cpu(),
                    "continuous_loss": per_sample_losses["continuous_loss"].detach().cpu(),
                    "binary_loss": per_sample_losses["binary_loss"].detach().cpu(),
                }
            )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Dict with "X".
            batch_idx: Index of current batch.
        """
        loss, total_mask, _ = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "test/mask_ratio",
            total_mask.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

    # -- Per-sample residual storage hooks --

    def on_train_epoch_end(self) -> None:
        """Flush accumulated training residuals to disk."""
        if self._store_residuals and self._train_residuals:
            self._flush_residuals("train", self._train_residuals)
            self._train_residuals.clear()

    def on_validation_epoch_end(self) -> None:
        """Flush accumulated validation residuals to disk."""
        if self._store_residuals and self._val_residuals:
            self._flush_residuals("val", self._val_residuals)
            self._val_residuals.clear()

    def _flush_residuals(self, split: str, residuals: list[dict[str, torch.Tensor]]) -> None:
        """Concatenate batch residuals and save to disk.

        Args:
            split: "train" or "val".
            residuals: List of per-batch dicts with "idx", "continuous_loss", "binary_loss".
        """
        merged = {
            "idx": torch.cat([r["idx"] for r in residuals]),
            "continuous_loss": torch.cat([r["continuous_loss"] for r in residuals]),
            "binary_loss": torch.cat([r["binary_loss"] for r in residuals]),
            "epoch": self.current_epoch,
        }
        out_dir = self._residuals_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"epoch_{self.current_epoch:04d}.pt"
        torch.save(merged, path)

    def configure_optimizers(self) -> OptimizerLRScheduler | OptimizerLRSchedulerConfig:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dict with optimizer and lr_scheduler config.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

        # If lr_scheduler config dict is provided, use it directly
        if self.hparams.lr_scheduler is not None:
            return self._configure_from_lr_scheduler_config(optimizer)

        # Otherwise use scheduler string param
        scheduler = self.hparams.scheduler

        if scheduler == "constant":
            return {"optimizer": optimizer}

        elif scheduler == "cosine":
            return self._configure_cosine_scheduler(optimizer)

        elif scheduler == "exponential":
            return self._configure_exponential_scheduler(optimizer)

        else:
            raise ValueError(
                f"Unknown scheduler: {scheduler}. Use 'cosine', 'exponential', or 'constant'."
            )

    def _configure_from_lr_scheduler_config(self, optimizer: AdamW) -> OptimizerLRSchedulerConfig:
        """Configure scheduler from lr_scheduler config dict."""
        config = self.hparams.lr_scheduler

        # Build init_args, injecting runtime values
        init_args = dict(config.get("init_args", {}))
        class_path = config["class_path"]

        # Extract scheduler wrapper config
        step_based_schedulers = ("OneCycleLR", "CyclicLR")
        default_interval = (
            "step" if any(s in class_path for s in step_based_schedulers) else "epoch"
        )
        interval = init_args.pop("interval", default_interval)
        frequency = init_args.pop("frequency", 1)
        monitor = init_args.pop("monitor", "val/loss")

        # Auto-inject T_max for CosineAnnealingLR if not specified
        if "CosineAnnealing" in class_path and "T_max" not in init_args:
            max_epochs = self.trainer.max_epochs
            if isinstance(max_epochs, int):
                init_args["T_max"] = max_epochs

        # Instantiate scheduler
        scheduler = instantiate_class(
            optimizer,
            {"class_path": class_path, "init_args": init_args},
        )

        lr_scheduler_config: LRSchedulerConfigType = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }

        if "ReduceLROnPlateau" in class_path:
            lr_scheduler_config["monitor"] = monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def _configure_cosine_scheduler(self, optimizer: AdamW) -> OptimizerLRSchedulerConfig:
        """Configure cosine annealing scheduler with warmup."""
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            return {"optimizer": optimizer}

        warmup_steps = max(1, int(self.hparams.warmup_ratio * total_steps))

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_steps
        )

        remaining_steps = total_steps - warmup_steps
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_steps, eta_min=self.hparams.eta_min
        )

        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }

    def _configure_exponential_scheduler(self, optimizer: AdamW) -> OptimizerLRSchedulerConfig:
        """Configure exponential decay scheduler with warmup."""
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            return {"optimizer": optimizer}

        max_epochs = self.trainer.max_epochs
        steps_per_epoch = total_steps // max_epochs

        warmup_steps = max(1, int(self.hparams.warmup_ratio * total_steps))

        # Default gamma: 0.995 per epoch
        gamma_epoch = 0.995
        gamma_step = gamma_epoch ** (1.0 / steps_per_epoch)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_steps
        )

        decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_step)

        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }
