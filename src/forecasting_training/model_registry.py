"""PyPOTS forecasting model factory.

Maps a :class:`ModelConfig` to a configured PyPOTS forecasting model instance.
Supports the three trainable PyPOTS forecasters benchmarked in OpenMHC:
DLinear, MixLinear, SegRNN. Each helper lazy-imports its PyPOTS class so the
top-level import stays cheap. The constructor kwargs mirror the eval-side
adapters in ``forecasting_evaluation/models/deep_learning_model/`` so a trained
checkpoint reloads with the same architecture.

All three models honor ``training.optimizer_lr`` via ``Adam(lr=...)`` (the
installed PyPOTS build accepts an ``optimizer`` kwarg for MixLinear as well —
the eval adapter omits it only because the optimizer is irrelevant at inference
time).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from forecasting_training.config import ModelConfig, OutputConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _apply_grad_clipping(model: Any, max_norm: float) -> None:
    """Wrap ``model.optimizer.step`` to clip gradients before each step.

    PyPOTS doesn't expose this natively. Called only when
    ``training.clip_grad_norm`` is set.
    """
    original_step = model.optimizer.step

    def _clipped_step(closure=None):
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm)
        return original_step(closure)

    model.optimizer.step = _clipped_step
    logger.info("Gradient clipping enabled: max_norm=%s", max_norm)


def create_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_config: OutputConfig,
) -> Any:
    """Instantiate an un-trained PyPOTS forecasting model from configuration.

    Args:
        model_config: Architecture choice and per-model hyperparameters.
        training_config: Optimizer, batch size, epochs, device.
        output_config: Where PyPOTS writes its mid-training ``.pypots`` file.

    Returns:
        Configured PyPOTS forecasting model (random init — the caller drives
        ``model.fit(...)`` next).
    """
    name = model_config.model_name.lower()

    if name == "dlinear":
        model = _create_dlinear(model_config, training_config, output_config)
    elif name == "mixlinear":
        model = _create_mixlinear(model_config, training_config, output_config)
    elif name == "segrnn":
        model = _create_segrnn(model_config, training_config, output_config)
    else:
        raise ValueError(f"Unsupported model {name!r}. Supported: dlinear, mixlinear, segrnn")

    if training_config.clip_grad_norm is not None and hasattr(model, "optimizer"):
        _apply_grad_clipping(model, training_config.clip_grad_norm)

    return model


def _shared_kwargs(model_config, training_config, output_config) -> dict[str, Any]:
    """Kwargs common to every PyPOTS forecasting model constructor."""
    from pypots.nn.modules.loss import MAE

    return dict(
        n_steps=model_config.n_steps,
        n_features=model_config.n_features,
        n_pred_steps=model_config.n_pred_steps,
        n_pred_features=model_config.n_features,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        patience=training_config.patience,
        training_loss=MAE,
        validation_metric=MAE,
        num_workers=training_config.num_workers,
        device=training_config.device,
        saving_path=output_config.saving_path,
        model_saving_strategy=output_config.model_saving_strategy,
    )


def _create_dlinear(model_config, training_config, output_config):
    from pypots.forecasting import DLinear
    from pypots.optim import Adam

    logger.info(
        "Creating DLinear forecaster: n_steps=%d, n_pred_steps=%d, n_features=%d, "
        "moving_avg_window_size=%d, individual=%s",
        model_config.n_steps,
        model_config.n_pred_steps,
        model_config.n_features,
        model_config.moving_avg_window_size,
        model_config.individual,
    )
    kwargs = _shared_kwargs(model_config, training_config, output_config)
    kwargs.update(
        moving_avg_window_size=model_config.moving_avg_window_size,
        individual=model_config.individual,
        optimizer=Adam(lr=training_config.optimizer_lr),
    )
    # DLinear only accepts d_model when not in individual-channel mode.
    if not model_config.individual:
        kwargs["d_model"] = model_config.d_model
    return DLinear(**kwargs)


def _create_mixlinear(model_config, training_config, output_config):
    from pypots.forecasting import MixLinear
    from pypots.optim import Adam

    logger.info(
        "Creating MixLinear forecaster: n_steps=%d, n_pred_steps=%d, n_features=%d, "
        "period_len=%d, lpf=%d, alpha=%s, rank=%d",
        model_config.n_steps,
        model_config.n_pred_steps,
        model_config.n_features,
        model_config.period_len,
        model_config.lpf,
        model_config.alpha,
        model_config.rank,
    )
    kwargs = _shared_kwargs(model_config, training_config, output_config)
    kwargs.update(
        period_len=model_config.period_len,
        lpf=model_config.lpf,
        alpha=model_config.alpha,
        rank=model_config.rank,
        optimizer=Adam(lr=training_config.optimizer_lr),
    )
    return MixLinear(**kwargs)


def _create_segrnn(model_config, training_config, output_config):
    from pypots.forecasting import SegRNN
    from pypots.optim import Adam

    logger.info(
        "Creating SegRNN forecaster: n_steps=%d, n_pred_steps=%d, n_features=%d, "
        "seg_len=%d, d_model=%d, dropout=%s",
        model_config.n_steps,
        model_config.n_pred_steps,
        model_config.n_features,
        model_config.seg_len,
        model_config.d_model,
        model_config.dropout,
    )
    kwargs = _shared_kwargs(model_config, training_config, output_config)
    kwargs.update(
        seg_len=model_config.seg_len,
        d_model=model_config.d_model,
        dropout=model_config.dropout,
        optimizer=Adam(lr=training_config.optimizer_lr),
    )
    return SegRNN(**kwargs)
