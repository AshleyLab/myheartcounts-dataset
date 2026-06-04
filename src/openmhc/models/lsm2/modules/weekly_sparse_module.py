"""Lightning module for Weekly Sparse Decoder MAE pretraining."""

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

from openmhc.models.lsm2 import create_inherited_mask
from openmhc.models.lsm2.weekly_sparse_decoder import WeeklySparseDecoderLSM2

__all__ = ["WeeklySparseLSM2Module"]


class WeeklySparseLSM2Module(pl.LightningModule):
    """Lightning module for weekly MAE with sparse cross-day decoder.

    Wraps WeeklySparseDecoderLSM2 with training/validation logic identical
    to MAEModule. Input: (B, 19, 10080) weekly tensors from
    MAEWeeklyMinuteDataModule.

    Args:
        seq_length: Per-day sequence length (default: 1440).
        patch_size: Patch size in minutes (default: 10).
        in_channels: Number of sensor channels (default: 19).
        embed_dim: Encoder embedding dimension.
        depth: Encoder transformer depth.
        num_heads: Encoder attention heads.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Decoder layers (alternating local/cross).
        decoder_num_heads: Decoder attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: QKV bias.
        norm_pix_loss: Normalize pixel loss.
        dropout_removal_ratio: Token dropout ratio.
        mask_ratio: Artificial masking ratio.
        use_hybrid_loss: MSE+BCE hybrid loss.
        continuous_channels: Continuous channel indices.
        channel_weights: Per-channel loss weights.
        num_days: Days per week (default: 7).
        window_minutes: Cross-day window width (default: 120).
        daily_checkpoint_path: Path to daily MAE checkpoint for weight init.
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay.
        scheduler: LR scheduler type.
        warmup_ratio: Warmup fraction.
        eta_min: Cosine scheduler minimum LR.
        betas: Adam beta coefficients.
        lr_scheduler: Custom scheduler config dict.
        store_residuals: Enable per-sample residual storage.
        residuals_dir: Output directory for residuals.
    """

    def __init__(  # noqa: D107
        self,
        # Model parameters
        seq_length: int = 1440,
        patch_size: int = 10,
        in_channels: int = 19,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_pix_loss: bool = False,
        dropout_removal_ratio: float = 0.5,
        mask_ratio: float = 0.5,
        use_hybrid_loss: bool = True,
        continuous_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        channel_weights: list[float] | None = None,
        # Weekly sparse decoder params
        num_days: int = 7,
        window_minutes: int = 120,
        use_rope_day_embed: bool = False,
        # Day masking (training-time augmentation)
        day_mask_count: int = 0,
        reconstruct_masked_days: bool = False,
        # Weight init
        daily_checkpoint_path: str | None = None,
        freeze_encoder: bool = False,
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
        super().__init__()
        self.save_hyperparameters()

        self.model = WeeklySparseDecoderLSM2(
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
            num_days=num_days,
            window_minutes=window_minutes,
            use_rope_day_embed=use_rope_day_embed,
        )

        self.patch_size = patch_size

        if daily_checkpoint_path:
            # The private MHC-benchmark training repo provides
            # ``utils.wandb_artifact`` to resolve ``wandb:<artifact>`` refs into
            # a local checkpoint path before pre-loading the daily encoder.
            # In the public openmhc package that helper isn't available — and
            # it isn't needed for inference: Lightning's ``load_from_checkpoint``
            # re-runs ``__init__`` with the saved ``daily_checkpoint_path``
            # hparam (which may be a ``wandb:`` reference), then immediately
            # overwrites the model state with ``checkpoint["state_dict"]``. So
            # the pre-load below is redundant at inference and we can safely
            # skip it when the private resolver isn't importable.
            try:
                from utils.wandb_artifact import (
                    is_wandb_reference,
                    resolve_checkpoint_path,
                )

                if is_wandb_reference(daily_checkpoint_path):
                    daily_checkpoint_path = str(
                        resolve_checkpoint_path(daily_checkpoint_path)
                    )
                self.model.load_daily_encoder_weights(daily_checkpoint_path)
            except ImportError:
                pass

        if freeze_encoder:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.model.pos_embed.requires_grad = False

        self._store_residuals = store_residuals
        self._residuals_dir = Path(residuals_dir) if residuals_dir else None
        self._train_residuals: list[dict[str, torch.Tensor]] = []
        self._val_residuals: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        x: torch.Tensor,
        inherited_mask: torch.Tensor | None = None,
        return_per_sample: bool = False,
        day_offsets: torch.Tensor | None = None,
        original_target: torch.Tensor | None = None,
        day_recon_patch_mask: torch.Tensor | None = None,
    ):
        """Forward pass through the model."""
        return self.model(
            x, inherited_mask, return_per_sample=return_per_sample, day_offsets=day_offsets,
            original_target=original_target, day_recon_patch_mask=day_recon_patch_mask,
        )

    def _apply_day_masking(
        self,
        x: torch.Tensor,
        num_real_days: torch.Tensor | None,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask ``count`` random days per sample by setting them to NaN.

        Only applies to samples with all 7 real days present.

        Returns:
            (masked_x, day_mask) where day_mask is (B, num_days) with 1 for masked days.
        """
        B, C, L = x.shape
        seq_len = self.hparams.seq_length
        num_days = self.hparams.num_days
        x = x.clone()
        x_4d = x.view(B, C, num_days, seq_len)

        eligible = torch.ones(B, dtype=torch.bool, device=x.device)
        if num_real_days is not None:
            eligible = num_real_days.to(x.device) == num_days

        day_mask = torch.zeros(B, num_days, device=x.device)
        if not eligible.any():
            return x, day_mask

        noise = torch.rand(B, num_days, device=x.device)
        _, indices = noise.topk(count, dim=1)
        day_mask.scatter_(1, indices, 1.0)
        day_mask = day_mask * eligible.float().unsqueeze(1)

        x_4d.masked_fill_(day_mask[:, None, :, None].bool(), float("nan"))
        return x, day_mask

    def _day_mask_to_patch_mask(self, day_mask: torch.Tensor) -> torch.Tensor:
        """Convert (B, num_days) day mask to (B, num_patches) patch mask."""
        B = day_mask.shape[0]
        num_days = self.hparams.num_days
        patches_per_channel_per_day = self.hparams.seq_length // self.patch_size
        # Build (num_days,) → (C, num_days, Pd) → (C * num_days * Pd,)
        # day_mask is (B, num_days) → (B, C, num_days, Pd)
        patch_mask = day_mask[:, None, :, None].expand(
            B, self.hparams.in_channels, num_days, patches_per_channel_per_day
        )
        # Reshape to channel-major patch order: (B, C, num_days * Pd) → (B, C * num_days * Pd)
        patch_mask = patch_mask.reshape(B, -1)
        return patch_mask

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        x = batch["X"]  # (B, C, num_days * seq_length) with NaN
        original_target = None
        day_recon_patch_mask = None

        # Natural inherited mask: marks patches whose ground truth is NaN
        # BEFORE artificial day masking. Used to gate the day-reconstruction
        # loss so we don't supervise patches whose only "target" is
        # nan_to_num(NaN)=0. Must be computed on the original x; computing
        # it after _apply_day_masking would let it absorb the artificial
        # NaNs and zero out the entire day-recon supervision.
        natural_inherited_mask = create_inherited_mask(x, self.patch_size)

        if self.training and self.hparams.day_mask_count > 0:
            if self.hparams.reconstruct_masked_days:
                original_target = torch.nan_to_num(x, nan=0.0)
            x, day_mask = self._apply_day_masking(
                x, batch.get("num_real_days"), self.hparams.day_mask_count
            )
            if self.hparams.reconstruct_masked_days and day_mask.any():
                raw_patch_mask = self._day_mask_to_patch_mask(day_mask)
                # Pre-gate by natural mask: only supervise day-masked patches
                # that had real (non-NaN) ground truth.
                day_recon_patch_mask = raw_patch_mask * (1 - natural_inherited_mask)

        # Encoder-input mask: natural NaNs ∪ day-masked NaNs. The encoder
        # drops both so the model is forced to reconstruct the masked days
        # rather than copy zeros (post-nan_to_num) through.
        inherited_mask = create_inherited_mask(x, self.patch_size)
        x_filled = torch.nan_to_num(x, nan=0.0)
        day_offsets = batch.get("day_offsets")

        if self._store_residuals:
            loss, pred, total_mask, per_sample_losses, day_recon_loss = self(
                x_filled, inherited_mask, return_per_sample=True, day_offsets=day_offsets,
                original_target=original_target, day_recon_patch_mask=day_recon_patch_mask,
            )
            return loss, total_mask, per_sample_losses, day_recon_loss
        else:
            loss, pred, total_mask, day_recon_loss = self(
                x_filled, inherited_mask, day_offsets=day_offsets,
                original_target=original_target, day_recon_patch_mask=day_recon_patch_mask,
            )
            return loss, total_mask, None, day_recon_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, total_mask, per_sample_losses, day_recon_loss = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs,
        )
        self.log(
            "train/mask_ratio", total_mask.mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        if day_recon_loss is not None:
            self.log(
                "train/loss_day_recon", day_recon_loss,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
            )

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
        """Validation step."""
        loss, total_mask, per_sample_losses, day_recon_loss = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log(
            "val/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs,
        )
        self.log(
            "val/mask_ratio", total_mask.mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        if day_recon_loss is not None:
            self.log(
                "val/loss_day_recon", day_recon_loss,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
            )

        if self._store_residuals and per_sample_losses is not None:
            self._val_residuals.append(
                {
                    "idx": batch["idx"].detach().cpu(),
                    "continuous_loss": per_sample_losses["continuous_loss"].detach().cpu(),
                    "binary_loss": per_sample_losses["binary_loss"].detach().cpu(),
                }
            )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, total_mask, _, day_recon_loss = self._shared_step(batch)

        bs = batch["X"].size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(
            "test/mask_ratio", total_mask.mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        if day_recon_loss is not None:
            self.log(
                "test/loss_day_recon", day_recon_loss,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=bs,
            )

    # -- Per-sample residual hooks --

    def on_train_epoch_end(self) -> None:
        """Flush training residuals."""
        if self._store_residuals and self._train_residuals:
            self._flush_residuals("train", self._train_residuals)
            self._train_residuals.clear()

    def on_validation_epoch_end(self) -> None:
        """Flush validation residuals."""
        if self._store_residuals and self._val_residuals:
            self._flush_residuals("val", self._val_residuals)
            self._val_residuals.clear()

    def _flush_residuals(self, split: str, residuals: list[dict[str, torch.Tensor]]) -> None:
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

    # -- Optimizer / Scheduler --

    def configure_optimizers(self) -> OptimizerLRScheduler | OptimizerLRSchedulerConfig:
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

        if self.hparams.lr_scheduler is not None:
            return self._configure_from_lr_scheduler_config(optimizer)

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
        config = self.hparams.lr_scheduler

        init_args = dict(config.get("init_args", {}))
        class_path = config["class_path"]

        step_based_schedulers = ("OneCycleLR", "CyclicLR")
        default_interval = (
            "step" if any(s in class_path for s in step_based_schedulers) else "epoch"
        )
        interval = init_args.pop("interval", default_interval)
        frequency = init_args.pop("frequency", 1)
        monitor = init_args.pop("monitor", "val/loss")

        if "CosineAnnealing" in class_path and "T_max" not in init_args:
            max_epochs = self.trainer.max_epochs
            if isinstance(max_epochs, int):
                init_args["T_max"] = max_epochs

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
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            return {"optimizer": optimizer}

        max_epochs = self.trainer.max_epochs
        steps_per_epoch = total_steps // max_epochs

        warmup_steps = max(1, int(self.hparams.warmup_ratio * total_steps))

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
