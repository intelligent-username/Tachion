"""Given a dataframe for a company, write the dates in a clear, standardized way that DeepAR can use"""

import pandas as pd


def add_date_features(df: pd.DataFrame, date_col: str = 'datetime') -> pd.DataFrame:
    """
    Add known date-based covariates: day_of_week, day_of_month, quarter.
    Assumes `date_col` is already datetime.
    """
    df = df.copy()
    df['day_of_week'] = df[date_col].dt.dayofweek        # 0=Monday, 6=Sunday
    df['day_of_month'] = df[date_col].dt.day             # 1-31
    df['quarter'] = df[date_col].dt.quarter              # 1-4
    return df
