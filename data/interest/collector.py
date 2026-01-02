"""
Interest Rate and Macroeconomic Indicators Collector

Collects data from two sources:
1. FRED (Federal Reserve Economic Data) - for economic indicators like:
   - PCEPILFE: Personal Consumption Expenditures (inflation measure)
   - UNRATE: Unemployment Rate
   - NROU: Noncyclical Rate of Unemployment  
   - GS3M: 3-Month Treasury yield
   - GS2: 2-Year Treasury yield
   - GS10: 10-Year Treasury yield
   - NFCI: National Financial Conditions Index
   - DFEDTARU: Federal Funds Target Rate Upper Bound

Also computes:
   - Spread_3M_2Y = GS3M - GS2
   - Spread_2Y_10Y = GS2 - GS10
"""

import json
from pathlib import Path
import pandas as pd
from core import call_specific_fred


def collect_fred_data(series_ids):
    path = Path(__file__).resolve().parent / "raw"
    path.mkdir(parents=True, exist_ok=True)

    call_specific_fred(str(path), series_ids=series_ids)

    # JSON-compatible yield spread computation
    gs3m_file = path / "GS3M.json"
    gs2_file = path / "GS2.json"
    gs10_file = path / "GS10.json"

    if all(f.exists() for f in [gs3m_file, gs2_file, gs10_file]):
        with gs3m_file.open("r") as f:
            df_3m = pd.DataFrame(json.load(f))
            df_3m['DATE'] = pd.to_datetime(df_3m['datetime'])
            df_3m.set_index('DATE', inplace=True)
            df_3m.rename(columns={'value':'GS3M'}, inplace=True)

        with gs2_file.open("r") as f:
            df_2y = pd.DataFrame(json.load(f))
            df_2y['DATE'] = pd.to_datetime(df_2y['datetime'])
            df_2y.set_index('DATE', inplace=True)
            df_2y.rename(columns={'value':'GS2'}, inplace=True)

        with gs10_file.open("r") as f:
            df_10y = pd.DataFrame(json.load(f))
            df_10y['DATE'] = pd.to_datetime(df_10y['datetime'])
            df_10y.set_index('DATE', inplace=True)
            df_10y.rename(columns={'value':'GS10'}, inplace=True)

        # Align on dates
        df = pd.concat([df_3m, df_2y, df_10y], axis=1, join='inner')

        # Compute spreads
        df['Spread_3M_2Y'] = df['GS3M'] - df['GS2']
        df['Spread_2Y_10Y'] = df['GS2'] - df['GS10']

        # Save computed spreads
        df[['Spread_3M_2Y', 'Spread_2Y_10Y']].to_csv(path / "YieldCurveSpreads.csv")
        print("Yield curve spreads computed and saved.")



def collect():
    """
    Gather all interest rate and macro data.
    Reads tickers from fred_tickers.txt
    """
    # Read FRED tickers
    fred_tickers = []
    tickers_file = Path(__file__).resolve().parent / "fred_tickers.txt"
    with tickers_file.open("r") as f:
        for line in f:
            line = line.split("#")[0].strip()  # Strip inline comments
            if line:
                fred_tickers.append(line)

    print(f"Collecting {len(fred_tickers)} FRED series...")
    collect_fred_data(fred_tickers)


if __name__ == "__main__":
    print("Collecting interest rate and macroeconomic data...")
    collect()
