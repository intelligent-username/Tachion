"""
Train DeepAR model using GluonTS.

Loads processed parquet data, converts to GluonTS PandasDataset format,
trains DeepAREstimator, and saves the trained predictor.
"""

import argparse
from pathlib import Path

from gluonts.evaluation import make_evaluation_predictions, Evaluator

from .deep import create_deepar_estimator, save_predictor
from .loader import load_gluonts_dataset, get_asset_freq


def train(
    asset: str = "crypto",
    prediction_length: int = 24,
    context_length: int = 48,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "auto",
):
    """
    Train DeepAR model for specified asset type.
    
    :param asset: Asset type ('crypto', 'equities', 'forex', 'comm')
    :param prediction_length: Forecast horizon
    :param context_length: Historical context window
    :param hidden_size: LSTM hidden dimension
    :param num_layers: Number of LSTM layers
    :param batch_size: Training batch size
    :param epochs: Number of epochs
    :param lr: Learning rate
    :param val_split: Fraction for validation
    :param device: Device for training ('auto', 'cpu', 'cuda')
    """
    print(f"Training DeepAR for {asset}")
    print(f"Config: prediction_length={prediction_length}, context={context_length}, "
          f"hidden={hidden_size}, layers={num_layers}")
    
    # Get frequency for this asset type
    freq = get_asset_freq(asset)
    
    # Load data as GluonTS datasets
    print("\nLoading data...")
    train_ds, test_ds = load_gluonts_dataset(
        asset_type=asset,
        prediction_length=prediction_length,
        val_split=val_split,
    )
    
    print(f"Train series: {len(list(train_ds))}")
    print(f"Test series: {len(list(test_ds))}")
    
    # Create estimator
    estimator = create_deepar_estimator(
        prediction_length=prediction_length,
        freq=freq,
        context_length=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )
    
    # Train
    print("\nTraining...")
    predictor = estimator.train(train_ds)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    evaluator = Evaluator(quantiles=[0.025, 0.5, 0.975])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    
    print("\nAggregate Metrics:")
    for metric in ["CRPS", "ND", "RMSE", "MASE"]:
        if metric in agg_metrics:
            print(f"  {metric}: {agg_metrics[metric]:.4f}")
    
    # Save model
    models_dir = Path(__file__).resolve().parents[1] / "models" / f"deepar_{asset}"
    save_predictor(predictor, str(models_dir))
    print(f"\nModel saved to: {models_dir}")
    
    return predictor, agg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepAR model with GluonTS")
    parser.add_argument("--asset", type=str, default="crypto",
                        choices=["crypto", "equities", "forex", "comm"])
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    
    args = parser.parse_args()
    
    train(
        asset=args.asset,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
