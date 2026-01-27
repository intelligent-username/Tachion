"""
Temporal Fusion Transformer (TFT) model wrapper using GluonTS.
"""

from pathlib import Path
from typing import Optional, List

import torch
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from core.training.constants import (
    TFT_NUM_HEADS,
    TFT_HIDDEN_DIM,
    TFT_VARIABLE_DIM,
    TFT_DROPOUT_RATE,
    TFT_LEARNING_RATE,
    TFT_WEIGHT_DECAY,
    TFT_BATCH_SIZE,
    TFT_NUM_BATCHES_PER_EPOCH,
    TFT_EPOCHS,
    DEFAULT_DEVICE,
)


class DetailedProgressBar(TQDMProgressBar):
    """Custom progress bar that shows batch totals."""
    
    def __init__(self, num_batches_per_epoch: int):
        super().__init__(refresh_rate=1)
        self.num_batches = num_batches_per_epoch
    
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.num_batches
        return bar


def create_tft_estimator(
    prediction_length: int,
    freq: str = "1H",
    context_length: Optional[int] = None,
    num_heads: int = TFT_NUM_HEADS,
    hidden_dim: int = TFT_HIDDEN_DIM,
    variable_dim: int = TFT_VARIABLE_DIM,
    dropout_rate: float = TFT_DROPOUT_RATE,
    lr: float = TFT_LEARNING_RATE,
    weight_decay: float = TFT_WEIGHT_DECAY,
    batch_size: int = TFT_BATCH_SIZE,
    num_batches_per_epoch: int = TFT_NUM_BATCHES_PER_EPOCH,
    epochs: int = TFT_EPOCHS,
    quantiles: Optional[List[float]] = None,
    device: str = DEFAULT_DEVICE,
    checkpoint_dir: Optional[Path] = None,
) -> TemporalFusionTransformerEstimator:
    """
    Create a configured TFT Estimator.
    
    TFT uses attention mechanisms and can parallelize across time steps,
    making it significantly faster to train than DeepAR on GPU.
    """
    if context_length is None:
        context_length = prediction_length
    
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Device selection
    if device == "auto":
        is_cuda = torch.cuda.is_available()
        accelerator = "gpu" if is_cuda else "cpu"
        device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
        print(f"  Device: {accelerator.upper()} ({device_name})")
    else:
        accelerator = "gpu" if device.startswith("cuda") else "cpu"
        print(f"  Device: {accelerator.upper()} (Forced)")

    # Setup callbacks
    callbacks = []
    
    # Progress bar
    progress_bar = DetailedProgressBar(num_batches_per_epoch=num_batches_per_epoch)
    callbacks.append(progress_bar)
    
    # Checkpoint callback
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="tft-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
        print(f"  Checkpoints: {checkpoint_dir}")

    # Trainer config
    # NOTE: enable_checkpointing=False because we add our own ModelCheckpoint callback
    # GluonTS would otherwise add a second one, causing a conflict
    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": accelerator,
        "devices": 1,
        "enable_model_summary": True,
        "enable_checkpointing": False,  # We use our own callback
        "callbacks": callbacks,
        "enable_progress_bar": True,
        "log_every_n_steps": 10,
        "limit_train_batches": num_batches_per_epoch,
    }
    
    return TemporalFusionTransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        quantiles=quantiles,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        variable_dim=variable_dim,
        dropout_rate=dropout_rate,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer_kwargs=trainer_kwargs,
    )
