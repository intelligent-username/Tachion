"""
Temporal Fusion Transformer using pytorch-forecasting.

This is significantly faster than GluonTS TFT due to pre-batched data pipelines.
"""

from pathlib import Path
from typing import Optional, List

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from core.training.constants import (
    TFT_NUM_HEADS,
    TFT_HIDDEN_DIM,
    TFT_DROPOUT_RATE,
    TFT_LEARNING_RATE,
    TFT_EPOCHS,
    DEFAULT_DEVICE,
)
from core.training.progress import CleanProgressBar


def create_tft_pf_model(
    training_dataset: TimeSeriesDataSet,
    hidden_size: int = TFT_HIDDEN_DIM,
    attention_head_size: int = TFT_NUM_HEADS,
    dropout: float = TFT_DROPOUT_RATE,
    lr: float = TFT_LEARNING_RATE,
    quantiles: Optional[List[float]] = None,
) -> TemporalFusionTransformer:
    """
    Create a pytorch-forecasting TFT model from a dataset.
    """
    if quantiles is None:
        quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        loss=QuantileLoss(quantiles=quantiles),
        reduce_on_plateau_patience=4,
        optimizer="AdamW",
    )
    
    return model


def create_trainer(
    epochs: int = TFT_EPOCHS,
    device: str = DEFAULT_DEVICE,
    checkpoint_dir: Optional[Path] = None,
    gradient_clip_val: float = 0.1,
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer for TFT.
    
    Uses float32 precision to avoid overflow in attention masks.
    """
    # Device selection
    if device == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = "gpu" if device.startswith("cuda") else "cpu"
    
    callbacks = [
        CleanProgressBar(),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ]
    
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
        )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        enable_model_summary=False,
        log_every_n_steps=50,
        enable_progress_bar=False,
        logger=False,
    )
    
    return trainer


class TFTPFPredictor:
    """
    Predictor wrapper to match GluonTS predictor interface.
    """
    
    def __init__(self, model: TemporalFusionTransformer, training_dataset: TimeSeriesDataSet):
        self.model = model
        self.training_dataset = training_dataset
        self.prediction_length = training_dataset.max_prediction_length
        self.quantiles = model.loss.quantiles
    
    def predict(self, dataloader) -> dict:
        """
        Generate predictions from a DataLoader.
        
        Returns dict with 'mean', 'lower', 'upper' arrays.
        """
        # Get predictions
        predictions = self.model.predict(dataloader, return_x=False)
        
        # Extract quantiles (assume 0.025, 0.5, 0.975 for 95% CI)
        q_idx_lower = 0  # 2.5%
        q_idx_median = len(self.quantiles) // 2  # 50%
        q_idx_upper = -1  # 97.5%
        
        return {
            "mean": predictions[:, :, q_idx_median].numpy(),
            "lower": predictions[:, :, q_idx_lower].numpy(),
            "upper": predictions[:, :, q_idx_upper].numpy(),
            "all_quantiles": predictions.numpy(),
        }
    
    def save(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_hparams": self.model.hparams,
        }, path / "model.pt")
    
    @classmethod
    def load(cls, path: Path, training_dataset: TimeSeriesDataSet):
        """Load model from checkpoint."""
        checkpoint = torch.load(path / "model.pt")
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **checkpoint["model_hparams"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, training_dataset)
