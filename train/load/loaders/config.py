"""Asset-specific configurations for data loading."""

# Asset-specific configurations
ASSET_CONFIG = {
    "crypto": {
        "freq": "1h",
        "target_col": "log_return",
        "item_id_col": "symbol",
        # Only the target is unknown. Engineered covariates are known reals.
        "unknown_reals": [],
        "known_reals": [
            "log_return_lag1",
            "volume_change_lag1",
            "5_period_MA",
            "20_period_MA",
            "rolling_volatility_5_lag1",
            "rolling_volatility_20_lag1",
            "btc_log_return_lag1",
            "btc_log_return",
        ],
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
        "item_id_col": "ticker",
        "unknown_reals": [],
        "known_reals": [
            "log_return_lag1",
            "volume_change_lag1",
            "5_day_MA",
            "50_day_MA",
            "rolling_volatility_5_lag1",
            "rolling_volatility_50_lag1",
            "sp_log_return_lag1",
            "delta_vix_lag1",
            "sp_log_return",
        ],
        "known_categoricals": [
            "day_of_week",
            "day_of_month",
            "quarter",
        ],
    },
    "forex": {
        "freq": "1h",
        "target_col": "log_return",
        "item_id_col": "symbol",
        "unknown_reals": [],
        "known_reals": [
            "log_return_lag1",
            "MA_50",
            "MA_200",
            "rolling_vol_50_lag1",
            "rolling_vol_200_lag1",
        ],
        "known_categoricals": [
            "day_of_week",
            "day_of_month",
            "quarter",
        ],
    },
    "comm": {
        "freq": "1D",
        "target_col": "log_return",
        "item_id_col": "symbol",
        "unknown_reals": [],
        "known_reals": [
            "log_return_lag1",
            "MA_50",
            "MA_200",
            "rolling_vol_50_lag1",
            "rolling_vol_200_lag1",
        ],
        "known_categoricals": [
            "day_of_week",
            "day_of_month",
            "quarter",
        ],
    },
}
