"""Rolling Volatility Calculator."""

import pandas as pd


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling standard deviation of `series` over `window` periods.
    Typically applied to log returns.
    """
    return series.rolling(window=window, min_periods=window).std()
