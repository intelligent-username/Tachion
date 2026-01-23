"""
Train DeepAR model for time series forecasting

Loads processed parquet data from data/{asset}/processed/
Trains DeepAR with Student-t likelihood
Saves model to models/
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .deep import DeepAR, student_t_nll


class TimeSeriesDataset(Dataset):
    """
    Dataset for DeepAR training.
    Creates sliding window sequences from processed data.
    """
    
    def __init__(self, df, feature_cols, target_col, context_length=48, prediction_length=1):
        """
        :param df: DataFrame with columns for features and target
        :param feature_cols: List of feature column names
        :param target_col: Target column name (e.g., 'log_return')
        :param context_length: Number of historical steps to use
        :param prediction_length: Number of steps to predict (for training, usually 1)
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Extract feature matrix and target
        self.features = df[feature_cols].values.astype(np.float32)
        self.target = df[target_col].values.astype(np.float32)
        
        # Valid indices (need enough history)
        self.valid_indices = np.arange(context_length, len(df) - prediction_length + 1)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        
        # Context window
        x = self.features[i - self.context_length:i]
        
        # Target (can be single step or multi-step)
        y = self.target[i:i + self.prediction_length]
        
        return torch.tensor(x), torch.tensor(y)


def get_feature_cols(asset):
    """Get feature columns for each asset type."""
    if asset == "crypto":
        return [
            'log_return_lag1', 'volume_change', '5_period_MA', '20_period_MA',
            'rolling_volatility_5', 'rolling_volatility_20', 'btc_log_return_lag1',
            'hour_of_day', 'day_of_week', 'day_of_month', 'is_weekend'
        ]
    elif asset == "equities":
        return [
            'log_return_lag1', 'volume_change', '5_day_MA', '50_day_MA',
            'rolling_volatility_5', 'rolling_volatility_50', 'sp_log_return_lag1',
            'delta_vix_lag1', 'day_of_week', 'day_of_month', 'quarter'
        ]
    elif asset in ["forex", "comm"]:
        return [
            'log_return_lag1', 'volume_change', '5_period_MA', '20_period_MA',
            'rolling_volatility_5', 'rolling_volatility_20',
            'day_of_week', 'day_of_month', 'quarter'
        ]
    else:
        raise ValueError(f"Unknown asset type: {asset}")


def load_data(asset):
    """Load processed data for asset type."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / asset / "processed"
    
    # Find parquet file
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No processed data found in {data_dir}\n"
            f"Please run: python -m data.{asset}.formatter"
        )
    
    df = pd.read_parquet(parquet_files[0])
    
    # Drop NaN rows
    feature_cols = get_feature_cols(asset)
    df = df.dropna(subset=feature_cols + ['log_return'])
    
    return df, feature_cols


def train(
    asset="crypto",
    context_length=48,
    hidden_size=64,
    num_layers=2,
    batch_size=64,
    epochs=20,
    lr=1e-3,
    val_split=0.1,
    device=None
):
    """
    Train DeepAR model for specified asset.
    
    :param asset: Asset type ('crypto', 'equities', 'forex', 'comm')
    :param context_length: Historical context window size
    :param hidden_size: LSTM hidden dimension
    :param num_layers: Number of LSTM layers
    :param batch_size: Training batch size
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :param val_split: Fraction for validation
    :param device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training DeepAR for {asset} on {device}")
    print(f"Config: context={context_length}, hidden={hidden_size}, layers={num_layers}")
    
    # Load data
    print("\nLoading data...")
    df, feature_cols = load_data(asset)
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    
    # Time-aware split
    n = len(df)
    val_idx = int(n * (1 - val_split))
    
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, feature_cols, 'log_return', context_length)
    val_dataset = TimeSeriesDataset(val_df, feature_cols, 'log_return', context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    input_size = len(feature_cols)
    model = DeepAR(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=3  # Student-t: mu, log_sigma, nu
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            params = model(x)
            
            # Use last timestep's params to predict next value
            last_params = params[:, -1, :]  # (batch, output_size)
            loss = student_t_nll(last_params, y.squeeze(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                params = model(x)
                last_params = params[:, -1, :]
                loss = student_t_nll(last_params, y.squeeze(-1))
                val_loss += loss.item() * x.size(0)
        
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            models_dir = Path(__file__).resolve().parents[1] / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"deepar_{asset}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': 3,
                    'context_length': context_length
                }
            }, model_path)
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DeepAR model")
    parser.add_argument("--asset", type=str, default="crypto", 
                        choices=["crypto", "equities", "forex", "comm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train(
        asset=args.asset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length,
        lr=args.lr
    )
