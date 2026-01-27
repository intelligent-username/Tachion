"""
Data loader for GluonTS training.

Reads Parquet files and converts them to GluonTS-compatible datasets
(PandasDataset/ListDataset format).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset


# Asset-specific configurations
ASSET_CONFIG = {
    "crypto": {
        "freq": "1H",
        "target_col": "log_return",
        "item_id_col": "symbol",  # If available
    },
    "equities": {
        "freq": "1D",
        "target_col": "log_return",
        "item_id_col": "symbol",
    },
    "forex": {
        "freq": "1H",
        "target_col": "log_return",
        "item_id_col": "symbol",
    },
    "comm": {
        "freq": "1D",
        "target_col": "log_return",
        "item_id_col": "symbol",
    },
}


def get_asset_freq(asset_type: str) -> str:
    """Get the frequency string for a given asset type."""
    if asset_type not in ASSET_CONFIG:
        raise ValueError(f"Unknown asset type: {asset_type}")
    return ASSET_CONFIG[asset_type]["freq"]


def get_asset_path(asset_type: str) -> Path:
    """Find the processed parquet file for a given asset type."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / asset_type / "processed"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    return parquet_files[0]


def load_parquet_as_dataframe(asset_type: str) -> pd.DataFrame:
    """Load parquet data for an asset type."""
    path = get_asset_path(asset_type)
    df = pd.read_parquet(path)
    
    # Ensure datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    
    return df


def load_gluonts_dataset(
    asset_type: str,
    prediction_length: int = 24,
    val_split: float = 0.1,
) -> Tuple[ListDataset, ListDataset]:
    """
    Load parquet data and convert to GluonTS ListDataset format.
    
    Returns train and test datasets where test includes the full series
    and train has the last prediction_length steps removed.
    
    :param asset_type: Asset type to load
    :param prediction_length: Forecast horizon (also determines test hold-out)
    :param val_split: Fraction of data to use for validation
    
    Returns a tuple of (train_dataset, test_dataset)
    """
    config = ASSET_CONFIG.get(asset_type)
    if config is None:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    freq = config["freq"]
    target_col = config["target_col"]
    
    df = load_parquet_as_dataframe(asset_type)
    
    # Drop NaN in target
    df = df.dropna(subset=[target_col])
    
    # Get unique series if we have multiple (e.g., multiple symbols)
    item_id_col = config.get("item_id_col")
    
    if item_id_col and item_id_col in df.columns:
        # Multiple time series
        series_list = []
        for item_id, group in df.groupby(item_id_col):
            group = group.sort_index()
            target = group[target_col].values
            start = group.index[0]
            series_list.append({
                "target": target,
                "start": pd.Period(start, freq=freq),
                "item_id": str(item_id),
            })
    else:
        # Single time series
        df = df.sort_index()
        target = df[target_col].values
        start = df.index[0]
        series_list = [{
            "target": target,
            "start": pd.Period(start, freq=freq),
            "item_id": "main",
        }]
    
    # Split: train has last prediction_length removed from each series
    train_data = []
    test_data = []
    
    for series in series_list:
        full_target = series["target"]
        n = len(full_target)
        
        # For very short series, skip
        if n < prediction_length * 2:
            continue
        
        # Test uses full series
        test_data.append({
            "target": full_target,
            "start": series["start"],
            "item_id": series["item_id"],
        })
        
        # Train removes the last prediction_length points
        train_data.append({
            "target": full_target[:-prediction_length],
            "start": series["start"],
            "item_id": series["item_id"],
        })
    
    train_ds = ListDataset(train_data, freq=freq)
    test_ds = ListDataset(test_data, freq=freq)
    
    return train_ds, test_ds


def load_pandas_dataset(
    asset_type: str,
    prediction_length: int = 24,
) -> PandasDataset:
    """
    Load data as a GluonTS PandasDataset.
    
    This is an alternative to ListDataset that works directly with pandas.
    
    :param asset_type: Asset type to load
    :param prediction_length: Forecast horizon
    
    Returns a PandasDataset
    """
    config = ASSET_CONFIG.get(asset_type)
    if config is None:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    freq = config["freq"]
    target_col = config["target_col"]
    item_id_col = config.get("item_id_col")
    
    df = load_parquet_as_dataframe(asset_type)
    df = df.dropna(subset=[target_col])
    
    # PandasDataset expects:
    # - DataFrame with DatetimeIndex
    # - target column specified
    # - optionally item_id for multiple series
    
    if item_id_col and item_id_col in df.columns:
        # For multiple series, we need the item_id as a column
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df = df.set_index("timestamp")
        
        return PandasDataset.from_long_dataframe(
            df,
            target=target_col,
            item_id=item_id_col,
            freq=freq,
        )
    else:
        # Single series
        return PandasDataset(
            {None: df[[target_col]]},
            target=target_col,
            freq=freq,
        )
