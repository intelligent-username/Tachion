

"""
Commodity prices
Calls the OANDA API to collect ~10 years of historical commodity data
for Gold, Silver, and Oil.
"""

import json
import os
import datetime
import time

from core import call_specific_oanda


def write_data_co(instruments):
    """
    Get ~175000 lines (35 API calls) worth of data for the given list of instruments
    Using 30 minute intervals (M30 granularity).
    The caller will ensure everything is written in chronological order.
    Writes to data/comm/raw/
    """
    path = os.path.join("data", "comm", "raw")
    os.makedirs(path, exist_ok=True)

    # 5000 candles per call
    # 15 years × 260 days × 23 hours × 2 (30-min) = ~180,000 candles
    # ~180,000 / 5000 = ~36 calls
    num_calls = 36

    call_specific_oanda(path, instruments=instruments, num_calls=num_calls)

    # JSON records are written chronologically (oldest to newest)


if __name__ == "__main__":
    # Commodity instruments from comms.txt

    print("Collecting commodity data...")

    with open("data/comm/comms.txt", "r") as f:
        commodities = []
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                commodities.append(line)

    print(f"Found {len(commodities)} commodities to collect: {commodities}")
    write_data_co(commodities)

