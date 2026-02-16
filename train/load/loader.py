"""
Data loader for GluonTS training.

Reads Parquet files and converts them to GluonTS-compatible datasets
with caching for fast multi-worker data loading.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset

# Try to import CachingDataset for in-memory caching
try:
    from gluonts.dataset.common import CachingDataset
    HAS_CACHING = True
except ImportError:
    HAS_CACHING = False


# Asset-specific configurations
ASSET_CONFIG = {
    "crypto": {
        "freq": "1h",
        "target_col": "log_return",
        "item_id_col": "symbol",
        # Lagged covariates (unknown in the future)
        "unknown_reals": [
            "volume_change",
            "5_period_MA",
            "20_period_MA",
            "rolling_volatility_5",
            "rolling_volatility_20",
        ],
        "known_reals": [],
        "known_categoricals": [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "is_weekend",
        ],
    },
    "equities": {
        "freq": "1D",
        "target_col": "log_return",
        "item_id_col": "symbol",
        "unknown_reals": [],
        "known_reals": [],
        "known_categoricals": [],
    },
    "forex": {
        "freq": "1h",
        "target_col": "log_return",
        "item_id_col": "symbol",
        "unknown_reals": [],
        "known_reals": [],
        "known_categoricals": [],
    },
    "comm": {
        "freq": "1D",
        "target_col": "log_return",
        "item_id_col": "symbol",
        "unknown_reals": [],
        "known_reals": [],
        "known_categoricals": [],
    },
}


def get_asset_freq(asset_type: str) -> str:
    """Get the frequency string for a given asset type."""
    if asset_type not in ASSET_CONFIG:
        raise ValueError(f"Unknown asset type: {asset_type}")
    return ASSET_CONFIG[asset_type]["freq"]


def get_asset_path(asset_type: str) -> Path:
    """Find the processed parquet file for a given asset type."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / asset_type / "processed"
    
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
    
    # Feature columns
    unknown_reals = [c for c in config.get("unknown_reals", []) if c in df.columns]
    known_reals = [c for c in config.get("known_reals", []) if c in df.columns]
    known_categoricals = [c for c in config.get("known_categoricals", []) if c in df.columns]

    # Pre-encode categoricals efficiently
    cat_maps = {}
    for c in known_categoricals:
        df[c] = df[c].astype('category')
        cat_maps[c] = df[c].cat.codes.values

    
    # Get unique series if we have multiple (e.g., multiple symbols)
    item_id_col = config.get("item_id_col")
    
    # Helper to extract features for a group
    def extract_features(group_df):
        # Known features (future known): reals + encoded categoricals
        # Shape: (num_features, length)
        feat_dynamic_real = []
        for c in known_reals:
            feat_dynamic_real.append(group_df[c].values)
        for c in known_categoricals:
            # We already have codes in cat_maps, but need to index by the group's index
            # Actually easier to just use the group's values since we converted to category
            feat_dynamic_real.append(group_df[c].cat.codes.values.astype(np.float32))
        
        # Unknown features (past known only): reals
        # Shape: (num_features, length)
        past_feat_dynamic_real = []
        for c in unknown_reals:
            past_feat_dynamic_real.append(group_df[c].values)
            
        return np.array(feat_dynamic_real, dtype=np.float32), np.array(past_feat_dynamic_real, dtype=np.float32)

    if item_id_col and item_id_col in df.columns:
        # Multiple time series
        series_list = []
        for item_id, group in df.groupby(item_id_col):
            group = group.sort_index()
            target = group[target_col].values
            start = group.index[0]
            
            fdr, pfdr = extract_features(group)
            
            entry = {
                "target": target,
                "start": pd.Period(start, freq=freq),
                "item_id": str(item_id),
            }
            if fdr.size > 0: entry["feat_dynamic_real"] = fdr
            if pfdr.size > 0: entry["past_feat_dynamic_real"] = pfdr
            series_list.append(entry)
    else:
        # Single time series
        df = df.sort_index()
        target = df[target_col].values
        start = df.index[0]
        
        fdr, pfdr = extract_features(df)
        
        entry = {
            "target": target,
            "start": pd.Period(start, freq=freq),
            "item_id": "main",
        }
        if fdr.size > 0: entry["feat_dynamic_real"] = fdr
        if pfdr.size > 0: entry["past_feat_dynamic_real"] = pfdr
        series_list.append(entry)
    
    # Split: train has last prediction_length removed from each series
    train_data = []
    test_data = []
    
    for series in series_list:
        full_target = series["target"]
        n = len(full_target)
        
        # For very short series, skip
        if n < prediction_length * 2:
            continue
        
        # Optimize: use contiguous float32 arrays for faster GPU transfer
        full_target_opt = np.ascontiguousarray(full_target, dtype=np.float32)
        train_target_opt = np.ascontiguousarray(full_target[:-prediction_length], dtype=np.float32)
        
        # Test uses full series
        test_data.append({
            "target": full_target_opt,
            "start": series["start"],
            "item_id": series["item_id"],
        })
        
        # Train removes the last prediction_length points
        # Slice features for train
        train_entry = {
            "target": train_target_opt,
            "start": series["start"],
            "item_id": series["item_id"],
        }
        test_entry = {
            "target": full_target_opt,
            "start": series["start"],
            "item_id": series["item_id"],
        }
        
        if "feat_dynamic_real" in series:
            fdr = series["feat_dynamic_real"]
            test_entry["feat_dynamic_real"] = fdr
            # Train needs full future features (if known)? Usually standard DeepAR uses future known features
            # But the 'target' is cut short. The features should match the target length + prediction_length?
            # Actually standard practice: provide full features, cut target.
            train_entry["feat_dynamic_real"] = fdr[:, :-prediction_length] # Cut to match target length for now to be safe
            
        if "past_feat_dynamic_real" in series:
            pfdr = series["past_feat_dynamic_real"]
            test_entry["past_feat_dynamic_real"] = pfdr
            train_entry["past_feat_dynamic_real"] = pfdr[:, :-prediction_length]

        test_data.append(test_entry)
        train_data.append(train_entry)
    
    train_ds = ListDataset(train_data, freq=freq)
    test_ds = ListDataset(test_data, freq=freq)
    
    # Wrap with CachingDataset if available - caches transformed data in memory
    if HAS_CACHING:
        train_ds = CachingDataset(train_ds)
        test_ds = CachingDataset(test_ds)
        print(f"  Dataset caching: ENABLED")
    
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



# Pytorch TimeSeries forecasting's data loader

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    HAS_PF = True
except ImportError:
    HAS_PF = False
    TimeSeriesDataSet = None
    GroupNormalizer = None


def load_pf_dataset(
    asset_type: str,
    prediction_length: int = 24,
    context_length: int = 48,
) -> Tuple:
    """
    Load data as pytorch-forecasting TimeSeriesDataSet.
    
    This creates pre-tensorized datasets that are significantly faster
    to train than GluonTS datasets.
    
    :param asset_type: Asset type to load
    :param prediction_length: Forecast horizon
    :param context_length: Encoder length (lookback window)
    
    Returns a tuple of (training_dataset, validation_dataset)
    """
    if not HAS_PF:
        raise ImportError(
            "pytorch-forecasting is not installed. "
            "Install with: pip install pytorch-forecasting"
        )
    
    config = ASSET_CONFIG.get(asset_type)
    if config is None:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    target_col = config["target_col"]
    item_id_col = config.get("item_id_col", "symbol")
    
    # Use existing loader
    df = load_parquet_as_dataframe(asset_type)
    
    # Clean data: replace infinity with NaN, then drop rows with NaN in any used column
    all_feature_cols = [target_col] + config.get("unknown_reals", []) + config.get("known_reals", []) + config.get("known_categoricals", [])
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    # Cast floats to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=all_feature_cols)
    
    # Prepare for TimeSeriesDataSet
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})
    
    # Add time index (required by pytorch-forecasting)
    if item_id_col in df.columns:
        df["time_idx"] = df.groupby(item_id_col).cumcount()
    else:
        df["time_idx"] = range(len(df))
        df[item_id_col] = "main"
    
    # Training cutoff: leave room for validation
    training_cutoff = df["time_idx"].max() - prediction_length
    
    print(f"  Samples: {len(df):,}")
    print(f"  Series: {df[item_id_col].nunique()}")
    
    # Determine which feature columns actually exist in the data
    unknown_reals = [c for c in config.get("unknown_reals", []) if c in df.columns]
    known_reals = [c for c in config.get("known_reals", []) if c in df.columns]
    known_categoricals = [c for c in config.get("known_categoricals", []) if c in df.columns]
    
    # Convert categoricals to strings (required for embeddings)
    for c in known_categoricals:
        df[c] = df[c].astype(str)
    
    # Target is always an unknown real
    all_unknown = [target_col] + unknown_reals
    # time_idx is always a known real
    all_known = ["time_idx"] + known_reals
    
    print(f"  Unknown reals: {all_unknown}")
    print(f"  Known reals: {all_known}")
    print(f"  Known categoricals: {known_categoricals}")
    
    # Create training dataset
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=[item_id_col],
        min_encoder_length=context_length // 2,
        max_encoder_length=context_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        static_categoricals=[item_id_col],
        time_varying_known_reals=all_known,
        time_varying_unknown_reals=all_unknown,
        time_varying_known_categoricals=known_categoricals,
        target_normalizer=GroupNormalizer(groups=[item_id_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Validation dataset uses same params as training
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )
    
    return training, validation

