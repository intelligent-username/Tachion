

"""
Interest Rate and Macroeconomic Indicators Collector

Collects data from two sources:
1. FRED (Federal Reserve Economic Data) - for economic indicators like:
   - PCEPILFE: Personal Consumption Expenditures (inflation measure)
   - UNRATE: Unemployment Rate
   - NROU: Noncyclical Rate of Unemployment  
   - T10Y2Y: 10-Year minus 2-Year Treasury spread (yield curve)
   - NFCI: National Financial Conditions Index
   - DFEDTARU: Federal Funds Target Rate Upper Bound

2. Yahoo Finance - for Fed Funds Futures (ZQ=F) implied rates
"""

import os
from core import call_specific_fred, call_specific_yf


def collect_fred_data(series_ids):
    """
    Collect FRED economic indicator data going back 15 years.
    FRED API returns all data in one call (no pagination needed).
    Writes to data/interest/raw/
    
    :param series_ids: List of FRED series IDs to collect
    """
    path = os.path.join("data", "interest", "raw")
    os.makedirs(path, exist_ok=True)

    call_specific_fred(path, series_ids=series_ids)


def collect_yahoo_data(symbols):
    """
    Collect Yahoo Finance data (e.g., Fed Funds Futures) going back 15 years.
    Uses daily intervals for consistency with FRED data.
    Writes to data/interest/raw/
    
    :param symbols: List of Yahoo Finance ticker symbols to collect
    """
    path = os.path.join("data", "interest", "raw")
    os.makedirs(path, exist_ok=True)

    call_specific_yf(path, symbols=symbols, interval="1d")


def collect():
    """
    Main collection function that gathers all interest rate and macro data.
    Reads tickers from fred_tickers.txt and yahoo_tickers.txt
    """
    # Read FRED tickers
    fred_tickers = []
    with open("data/interest/fred_tickers.txt", "r") as f:
        for line in f:
            line = line.split("#")[0].strip()  # Strip inline comments
            if line:
                fred_tickers.append(line)

    # Read Yahoo tickers
    yahoo_tickers = []
    with open("data/interest/yahoo_tickers.txt", "r") as f:
        for line in f:
            line = line.split("#")[0].strip()  # Strip inline comments
            if line:
                yahoo_tickers.append(line)

    print(f"Collecting {len(fred_tickers)} FRED series...")
    collect_fred_data(fred_tickers)

    print(f"\nCollecting {len(yahoo_tickers)} Yahoo Finance tickers...")
    collect_yahoo_data(yahoo_tickers)


if __name__ == "__main__":
    print("Collecting interest rate and macroeconomic data...")
    
    collect()
    
