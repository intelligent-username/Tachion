"""
Crypto prices
Calls the TwelveData API to collect information for the past ~15000 lines (about 5 years)
(or less if not available) for 100 randomly selected stocks.
BTC will be used as a lagging covariate for broader market crypto conditions.
"""

# Careful in the model definition to not let BTC leak future information into other cryptos

import json
import os
import datetime
import time

from core import call_specific

def write_data(symbols):
    """
    Get ~15000 lines (3 API calls) worth of data for the given list of symbols
    Using 30 minute intervals.
    The caller will ensure everything is written in chronological order.
    Writes to data/crypto/raw/
    """
    path = os.path.join("data", "crypto", "raw")
    os.makedirs(path, exist_ok=True)

    # This is for equities specifically, need ~15k
    num_calls = 3

    call_specific(path, symbols=symbols, num_calls = num_calls)

    # note that the JSON records are written chronologically from newest to oldest
    # In the feature engineering (CSVs), remember to read backwards

if __name__ == "__main__":
    # A total of 30 cryptocurrencies listed on Coinbase Exchange

    print("Collecting data...")

    with open("data/crypto/coins.txt", "r") as f:
        companies = [line.rstrip("\n") for line in f if line[0] != "#" and line != "\n"]
    
    write_data(companies)

    print("Finished collecting data.")
