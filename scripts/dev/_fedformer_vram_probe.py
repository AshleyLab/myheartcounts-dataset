"""Measure peak VRAM for one fwd+bwd step at the paper-checkpoint FEDformer config.

Uses the public PyPOTS ``FEDformer`` wrapper exactly the way openmhc's
training pipeline does (see imputation_training.model_registry._create_fedformer),
so the measured VRAM is what the real training run will see.

Mirrors W&B run MHC_Dataset/mhc-pypots-fedformer/runs/ouqezdi7 (paper bundle):
- model: d_model=512, n_heads=8, d_ffn=128, modes=32, n_layers=2,
         moving_avg_window_size=25, dropout=0.1, n_steps=1440, n_features=19
- training: batch_size=128
"""

import numpy as np
import torch
from pypots.imputation import FEDformer
from pypots.nn.modules.loss import MAE
from pypots.optim import Adam

torch.manual_seed(42)
np.random.seed(42)

dev_name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {dev_name} ({total:.1f} GB)")

BS = 128
N_STEPS = 1440
N_FEATURES = 19

model = FEDformer(
    n_steps=N_STEPS,
    n_features=N_FEATURES,
    n_layers=2,
    d_model=512,
    n_heads=8,
    d_ffn=128,
    moving_avg_window_size=25,
    dropout=0.1,
    version="Fourier",
    modes=32,
    mode_select="random",
    batch_size=BS,
    epochs=1,
    patience=0,
    training_loss=MAE,
    validation_metric=MAE,
    optimizer=Adam(lr=0.002),
    num_workers=0,
    device="cuda",
    saving_path=None,
)

# Synthetic dataset: 4 batches' worth of random data with binary mask.
N_SAMPLES = BS * 4
rng = np.random.default_rng(42)
X = rng.standard_normal((N_SAMPLES, N_STEPS, N_FEATURES)).astype(np.float32)
mask = rng.binomial(1, 0.7, size=(N_SAMPLES, N_STEPS, N_FEATURES)).astype(np.float32)
X[mask == 0] = np.nan

train_set = {"X": X}
val_set = {"X": X, "X_ori": X}

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

print(f"\nFitting FEDformer for 1 epoch at batch_size={BS} on {N_SAMPLES} samples ...")
model.fit(train_set=train_set, val_set=val_set)

torch.cuda.synchronize()
peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

n_params = sum(p.numel() for p in model.model.parameters())
print(f"\nResults at batch_size={BS}:")
print(f"  Model params:                  {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")
print(f"  Peak allocated (real step):    {peak_alloc:.3f} GB")
print(f"  Peak reserved (incl. cache):   {peak_reserved:.3f} GB")
print("\nRecommendation:")
print(f"  Min GPU memory to fit:                {peak_reserved:.1f} GB")
print(f"  Safe with 30% headroom:               {peak_alloc * 1.3:.1f} GB")
