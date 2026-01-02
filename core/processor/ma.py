"""Moving Average Calculator"""

import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Compute simple moving average over `window` periods.
    """
    return series.rolling(window=window, min_periods=window).mean()
