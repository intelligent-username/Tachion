"""
DeepAR model wrapper using GluonTS.

This module provides a simplified interface to the GluonTS DeepAREstimator
for probabilistic time series forecasting with Student-t distribution.
"""

from typing import List, Optional
from pathlib import Path

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.predictor import Predictor


def create_deepar_estimator(
    prediction_length: int,
    freq: str = "1H",
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 64,
    dropout_rate: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-8,
    batch_size: int = 64,
    num_batches_per_epoch: int = 100,
    epochs: int = 20,
    num_parallel_samples: int = 100,
    num_feat_dynamic_real: int = 0,
    device: str = "auto",
) -> DeepAREstimator:
    """
    Create a configured DeepAREstimator with Student-t distribution.
    
    :param prediction_length: Number of time steps to forecast
    :param freq: Frequency of the time series (e.g., 'H', 'D', 'W')
    :param context_length: Number of historical steps for context (default: prediction_length)
    :param num_layers: Number of RNN layers
    :param hidden_size: Hidden dimension of RNN 
    :param dropout_rate: Dropout rate for regularization
    :param lr: Learning rate
    :param weight_decay: Weight decay for regularization
    :param batch_size: Batch size for training
    :param num_batches_per_epoch: Number of batches per epoch
    :param epochs: Number of training epochs
    :param num_parallel_samples: Number of samples for probabilistic forecasts
    :param num_feat_dynamic_real: Number of dynamic real-valued features
    :param device: Device for training ('auto', 'cpu', 'cuda')
    :return: Configured DeepAREstimator
    """
    if context_length is None:
        context_length = prediction_length
    
    # Trainer kwargs for PyTorch Lightning
    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": device if device != "auto" else "auto",
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    estimator = DeepAREstimator(
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
        num_feat_dynamic_real=num_feat_dynamic_real,
        distr_output=StudentTOutput(),  # Student-t for heavy tails
        scaling=True,  # Auto-scaling for numerical stability
        trainer_kwargs=trainer_kwargs,
    )
    
    return estimator


def load_predictor(model_dir: str) -> Predictor:
    """
    Load a trained DeepAR predictor from disk.
    
    :param model_dir: Directory containing the saved predictor
    :return: Loaded Predictor object
    """
    return Predictor.deserialize(Path(model_dir))


def save_predictor(predictor: Predictor, model_dir: str) -> None:
    """
    Save a trained predictor to disk.
    
    :param predictor: Trained Predictor object
    :param model_dir: Directory to save the predictor
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    predictor.serialize(model_path)
