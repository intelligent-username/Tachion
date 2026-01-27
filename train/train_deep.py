"""
Train time series models (DeepAR or TFT) using GluonTS.

Usage:
    python -m train.train_deep [asset] [model] [-n]
    
Examples:
    python -m train.train_deep crypto tft -n    # Train TFT on crypto
    python -m train.train_deep forex deepar -n  # Train DeepAR on forex
"""

from pathlib import Path
import time
import sys
import traceback
import torch

# Optimize for Tensor Cores
torch.set_float32_matmul_precision('medium')

from gluonts.evaluation import make_evaluation_predictions, Evaluator

from .deep import create_deepar_estimator
from .tft import create_tft_estimator
from .loader import load_gluonts_dataset, get_asset_freq
from core import set_training_defaults, save_predictor
from core.training.constants import (
    # DeepAR
    DEEPAR_PREDICTION_LENGTH,
    DEEPAR_CONTEXT_LENGTH,
    DEEPAR_BATCH_SIZE,
    DEEPAR_NUM_BATCHES_PER_EPOCH,
    DEEPAR_EPOCHS,
    DEEPAR_LEARNING_RATE,
    # TFT
    TFT_PREDICTION_LENGTH,
    TFT_CONTEXT_LENGTH,
    TFT_BATCH_SIZE,
    TFT_NUM_BATCHES_PER_EPOCH,
    TFT_EPOCHS,
    TFT_LEARNING_RATE,
    # Shared
    DEFAULT_DEVICE,
    DEFAULT_ASSET,
    DEFAULT_MODEL,
)


def train(
    asset: str = DEFAULT_ASSET,
    model: str = DEFAULT_MODEL,
    prediction_length: int = None,
    context_length: int = None,
    batch_size: int = None,
    num_batches_per_epoch: int = None,
    epochs: int = None,
    lr: float = None,
    device: str = DEFAULT_DEVICE,
):
    """Train a time series model for specified asset type."""
    
    # Set model-specific defaults
    if model == "tft":
        prediction_length = prediction_length or TFT_PREDICTION_LENGTH
        context_length = context_length or TFT_CONTEXT_LENGTH
        batch_size = batch_size or TFT_BATCH_SIZE
        num_batches_per_epoch = num_batches_per_epoch or TFT_NUM_BATCHES_PER_EPOCH
        epochs = epochs or TFT_EPOCHS
        lr = lr or TFT_LEARNING_RATE
    else:  # deepar
        prediction_length = prediction_length or DEEPAR_PREDICTION_LENGTH
        context_length = context_length or DEEPAR_CONTEXT_LENGTH
        batch_size = batch_size or DEEPAR_BATCH_SIZE
        num_batches_per_epoch = num_batches_per_epoch or DEEPAR_NUM_BATCHES_PER_EPOCH
        epochs = epochs or DEEPAR_EPOCHS
        lr = lr or DEEPAR_LEARNING_RATE
    
    model_name = "TFT" if model == "tft" else "DeepAR"
    
    print(f"\n{'='*60}")
    print(f"  Training {model_name} for {asset.upper()}")
    print(f"{'='*60}")
    
    # Setup paths
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir = models_dir / f"{model}_{asset}"
    checkpoint_dir = models_dir / f"{model}_{asset}_checkpoints"
    
    # Get frequency for this asset type
    freq = get_asset_freq(asset)
    
    # Load data
    print("\nLoading data...")
    train_ds, test_ds = load_gluonts_dataset(
        asset_type=asset,
        prediction_length=prediction_length,
    )
    print(f"  Train series: {len(list(train_ds))}")
    
    # Create estimator based on model choice
    if model == "tft":
        estimator = create_tft_estimator(
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            lr=lr,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            epochs=epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
    else:  # deepar
        estimator = create_deepar_estimator(
            prediction_length=prediction_length,
            freq=freq,
            context_length=context_length,
            lr=lr,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            epochs=epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
    
    # Train
    print(f"\nStarting training ({epochs} epochs, {num_batches_per_epoch} batches/epoch)...\n")
    train_start = time.time()
    predictor = None
    
    try:
        # num_workers=0 avoids Windows multiprocessing spawn overhead
        predictor = estimator.train(train_ds, num_workers=0)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. No model saved.")
        return None, None
    
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    train_time = time.time() - train_start
    print(f"\nTraining complete! ({train_time/60:.2f} min)")
    
    # Evaluate
    print("\nEvaluating...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )
    
    evaluator = Evaluator(quantiles=[0.025, 0.5, 0.975])
    agg_metrics, _ = evaluator(list(ts_it), list(forecast_it))
    
    print(f"\n  CRPS: {agg_metrics.get('CRPS', 'N/A'):.6f}")
    print(f"  RMSE: {agg_metrics.get('RMSE', 'N/A'):.6f}")
    
    # Save
    save_predictor(predictor, str(final_model_dir))
    print(f"\nModel saved to: {final_model_dir}")
    
    return predictor, agg_metrics


def parse_args():
    """Parse command line arguments."""
    asset = DEFAULT_ASSET
    model = DEFAULT_MODEL
    skip_modify = False
    
    valid_assets = ["crypto", "equities", "forex", "comm", "interest"]
    valid_models = ["deepar", "tft"]
    
    args = sys.argv[1:]
    
    for arg in args:
        if arg.lower() == "-n":
            skip_modify = True
        elif arg.lower() in valid_assets:
            asset = arg.lower()
        elif arg.lower() in valid_models:
            model = arg.lower()
        else:
            print(f"Warning: Unrecognized argument '{arg}'")
    
    return asset, model, skip_modify


if __name__ == "__main__":
    asset, model, skip_modify = parse_args()
    
    # Use model-specific defaults
    if model == "tft":
        defaults = {
            "asset": asset,
            "model": model,
            "epochs": TFT_EPOCHS,
            "prediction_length": TFT_PREDICTION_LENGTH,
            "context_length": TFT_CONTEXT_LENGTH,
            "batch_size": TFT_BATCH_SIZE,
            "num_batches_per_epoch": TFT_NUM_BATCHES_PER_EPOCH,
            "lr": TFT_LEARNING_RATE,
            "device": DEFAULT_DEVICE,
        }
    else:
        defaults = {
            "asset": asset,
            "model": model,
            "epochs": DEEPAR_EPOCHS,
            "prediction_length": DEEPAR_PREDICTION_LENGTH,
            "context_length": DEEPAR_CONTEXT_LENGTH,
            "batch_size": DEEPAR_BATCH_SIZE,
            "num_batches_per_epoch": DEEPAR_NUM_BATCHES_PER_EPOCH,
            "lr": DEEPAR_LEARNING_RATE,
            "device": DEFAULT_DEVICE,
        }
    
    if skip_modify:
        config = defaults
        print(f"Training Config: {', '.join([f'{k}={v}' for k, v in config.items()])}\n")
    else:
        config = set_training_defaults(defaults)
    
    train(
        asset=config["asset"],
        model=config.get("model", model),
        prediction_length=config["prediction_length"],
        context_length=config["context_length"],
        num_batches_per_epoch=config["num_batches_per_epoch"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        device=config["device"],
    )
