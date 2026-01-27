"""
DeepAR model wrapper using GluonTS.
"""

from pathlib import Path
from typing import Optional

import torch
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from core.training.constants import (
    DEEPAR_NUM_LAYERS,
    DEEPAR_HIDDEN_SIZE,
    DEEPAR_DROPOUT_RATE,
    DEEPAR_LEARNING_RATE,
    DEEPAR_WEIGHT_DECAY,
    DEEPAR_BATCH_SIZE,
    DEEPAR_NUM_BATCHES_PER_EPOCH,
    DEEPAR_EPOCHS,
    DEEPAR_NUM_PARALLEL_SAMPLES,
    DEFAULT_DEVICE,
)


class DetailedProgressBar(TQDMProgressBar):
    """Custom progress bar that shows more stats and doesn't double-print."""
    
    def __init__(self, num_batches_per_epoch: int):
        super().__init__(refresh_rate=1)
        self.num_batches = num_batches_per_epoch
    
    def get_metrics(self, trainer, pl_module):
        # Get default metrics and add custom ones
        items = super().get_metrics(trainer, pl_module)
        # Remove redundant v_num
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.num_batches
        bar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return bar


def create_deepar_estimator(
    prediction_length: int,
    freq: str = "1H",
    context_length: Optional[int] = None,
    num_layers: int = DEEPAR_NUM_LAYERS,
    hidden_size: int = DEEPAR_HIDDEN_SIZE,
    dropout_rate: float = DEEPAR_DROPOUT_RATE,
    lr: float = DEEPAR_LEARNING_RATE,
    weight_decay: float = DEEPAR_WEIGHT_DECAY,
    batch_size: int = DEEPAR_BATCH_SIZE,
    num_batches_per_epoch: int = DEEPAR_NUM_BATCHES_PER_EPOCH,
    epochs: int = DEEPAR_EPOCHS,
    num_parallel_samples: int = DEEPAR_NUM_PARALLEL_SAMPLES,
    device: str = DEFAULT_DEVICE,
    checkpoint_dir: Optional[Path] = None,
) -> DeepAREstimator:
    """
    Create a configured DeepAREstimator with Student-t distribution.
    """
    if context_length is None:
        context_length = prediction_length
    
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
    
    # Custom progress bar with known total batches
    progress_bar = DetailedProgressBar(num_batches_per_epoch=num_batches_per_epoch)
    callbacks.append(progress_bar)
    
    # Checkpoint callback
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="deepar-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
        print(f"  Checkpoints: {checkpoint_dir}")

    # Trainer config
    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": accelerator,
        "devices": 1,
        "enable_model_summary": True,
        "enable_checkpointing": False,  # We use our own callback
        "callbacks": callbacks,
        "enable_progress_bar": True,
        # Log metrics every 10 batches for better visibility
        "log_every_n_steps": 10,
        # Limit train batches to show total in progress bar
        "limit_train_batches": num_batches_per_epoch,
    }
    
    return DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        num_parallel_samples=num_parallel_samples,
        distr_output=StudentTOutput(),
        scaling=True,
        trainer_kwargs=trainer_kwargs,
    )
