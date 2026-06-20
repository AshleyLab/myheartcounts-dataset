"""PyPOTS model factory: maps a :class:`ModelConfig` to a configured PyPOTS
model instance.

Supports the four neural imputers benchmarked in OpenMHC: BRITS, DLinear,
TimesNet, FEDformer. To add a new model, follow the pattern of the
existing ``_create_<name>`` helpers and extend the ``if/elif`` in
:func:`create_model`. Each helper lazy-imports its PyPOTS class so the
top-level import remains cheap.

The factory also wires optional gradient clipping (PyPOTS has no native
support) by monkey-patching ``model.optimizer.step``.

Ported (and trimmed to four models) from MHC-benchmark's
``src/pypots_training/model_registry.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from imputation_training.config import ModelConfig, OutputConfig, TrainingConfig

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
    """Instantiate a PyPOTS model from configuration.

    Args:
        model_config: Architecture choice and per-model hyperparameters.
        training_config: Optimizer, batch size, epochs, device.
        output_config: Where PyPOTS writes its mid-training ``.pypots`` file.

    Returns:
        Configured PyPOTS model (un-trained, weights are random init —
        the caller drives ``model.fit(...)`` next).
    """
    name = model_config.model_name.lower()

    if name == "brits":
        model = _create_brits(model_config, training_config, output_config)
    elif name == "dlinear":
        model = _create_dlinear(model_config, training_config, output_config)
    elif name == "timesnet":
        model = _create_timesnet(model_config, training_config, output_config)
    elif name == "fedformer":
        model = _create_fedformer(model_config, training_config, output_config)
    else:
        raise ValueError(
            f"Unsupported model {name!r}. "
            "Supported: brits, dlinear, timesnet, fedformer"
        )

    if training_config.clip_grad_norm is not None and hasattr(model, "optimizer"):
        _apply_grad_clipping(model, training_config.clip_grad_norm)

    return model


def _create_brits(model_config, training_config, output_config):
    from pypots.imputation import BRITS
    from pypots.nn.modules.loss import MAE
    from pypots.optim import Adam

    logger.info(
        "Creating BRITS model: n_steps=%d, n_features=%d, rnn_hidden_size=%d",
        model_config.n_steps,
        model_config.n_features,
        model_config.rnn_hidden_size,
    )
    return BRITS(
        n_steps=model_config.n_steps,
        n_features=model_config.n_features,
        rnn_hidden_size=model_config.rnn_hidden_size,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        patience=training_config.patience,
        training_loss=MAE,
        validation_metric=MAE,
        optimizer=Adam(
            lr=training_config.optimizer_lr,
            weight_decay=training_config.weight_decay,
        ),
        num_workers=training_config.num_workers,
        device=training_config.device,
        saving_path=output_config.saving_path,
        model_saving_strategy=output_config.model_saving_strategy,
    )


def _create_dlinear(model_config, training_config, output_config):
    from pypots.imputation import DLinear
    from pypots.nn.modules.loss import MAE
    from pypots.optim import Adam

    logger.info(
        "Creating DLinear model: n_steps=%d, n_features=%d, moving_avg_window_size=%d",
        model_config.n_steps,
        model_config.n_features,
        model_config.moving_avg_window_size,
    )
    kwargs = dict(
        n_steps=model_config.n_steps,
        n_features=model_config.n_features,
        moving_avg_window_size=model_config.moving_avg_window_size,
        individual=model_config.individual,
        ORT_weight=model_config.ORT_weight,
        MIT_weight=model_config.MIT_weight,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        patience=training_config.patience,
        training_loss=MAE,
        validation_metric=MAE,
        optimizer=Adam(
            lr=training_config.optimizer_lr,
            weight_decay=training_config.weight_decay,
        ),
        num_workers=training_config.num_workers,
        device=training_config.device,
        saving_path=output_config.saving_path,
        model_saving_strategy=output_config.model_saving_strategy,
    )
    # DLinear only accepts d_model when not in individual-channel mode.
    if not model_config.individual:
        kwargs["d_model"] = model_config.d_model
    return DLinear(**kwargs)


def _create_timesnet(model_config, training_config, output_config):
    from pypots.imputation import TimesNet
    from pypots.nn.modules.loss import MAE
    from pypots.optim import Adam

    logger.info(
        "Creating TimesNet model: n_steps=%d, n_features=%d, n_layers=%d, d_model=%d",
        model_config.n_steps,
        model_config.n_features,
        model_config.n_layers,
        model_config.d_model,
    )
    return TimesNet(
        n_steps=model_config.n_steps,
        n_features=model_config.n_features,
        n_layers=model_config.n_layers,
        top_k=model_config.top_k,
        d_model=model_config.d_model,
        d_ffn=model_config.d_ffn,
        n_kernels=model_config.n_kernels,
        dropout=model_config.dropout,
        apply_nonstationary_norm=model_config.apply_nonstationary_norm,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        patience=training_config.patience,
        training_loss=MAE,
        validation_metric=MAE,
        optimizer=Adam(
            lr=training_config.optimizer_lr,
            weight_decay=training_config.weight_decay,
        ),
        num_workers=training_config.num_workers,
        device=training_config.device,
        saving_path=output_config.saving_path,
        model_saving_strategy=output_config.model_saving_strategy,
    )


def _create_fedformer(model_config, training_config, output_config):
    """Construct a PyPOTS FEDformer.

    Important: the openmhc release-bundle layer (see
    :func:`imputation_training.release.write_release`) captures each
    ``FourierBlock.index`` after this constructor runs, so that
    inference can restore the same indices and avoid the upstream
    PyPOTS bug. The trainer MUST have called
    :func:`imputation_training.seeding.seed_everything` before this
    factory is invoked, otherwise the indices are still
    process-state-dependent (though the sidecar will still restore them
    correctly on load — the seed just makes the training run itself
    repeatable).
    """
    from pypots.imputation import FEDformer
    from pypots.nn.modules.loss import MAE
    from pypots.optim import Adam

    logger.info(
        "Creating FEDformer model: n_steps=%d, n_features=%d, n_layers=%d, "
        "d_model=%d, modes=%d, mode_select=%s",
        model_config.n_steps,
        model_config.n_features,
        model_config.n_layers,
        model_config.d_model,
        model_config.modes,
        model_config.mode_select,
    )
    return FEDformer(
        n_steps=model_config.n_steps,
        n_features=model_config.n_features,
        n_layers=model_config.n_layers,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        d_ffn=model_config.d_ffn,
        moving_avg_window_size=model_config.moving_avg_window_size,
        dropout=model_config.dropout,
        version=model_config.version,
        modes=model_config.modes,
        mode_select=model_config.mode_select,
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        patience=training_config.patience,
        training_loss=MAE,
        validation_metric=MAE,
        optimizer=Adam(
            lr=training_config.optimizer_lr,
            weight_decay=training_config.weight_decay,
        ),
        num_workers=training_config.num_workers,
        device=training_config.device,
        saving_path=output_config.saving_path,
        model_saving_strategy=output_config.model_saving_strategy,
    )
