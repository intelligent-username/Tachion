"""
Stock prices and S&P500 data collector
Calls the TwelveData API to collect information for the past ~5 years
(or less if not available) for 100 randomly selected stocks.
It then collects the past ~5 years of S&P500 index data.
All data is written into CSVs in the raw/ directory.
"""

import json
import os
import datetime
import time

from core import TwelveDataAPI

def write_data(symbols):
    """
    Get ~5 years (3 API calls) worth of data for the given list of symbols and S&P500 index
    Using 30 minute intervals
    """
    path = os.path.join("data", "equities", "raw")
    os.makedirs(path, exist_ok=True)

    # This is for equities specifically, need ~15k
    num_calls = 3

    call_specific(path, symbols=["SPY"], num_calls = num_calls)
    call_specific(path, symbols=symbols, num_calls = num_calls)

    # note that the JSON records are written chronologically from newest to oldest
    # In the feature engineering (CSVs), remember to read backwards


def call_specific(path, symbols, num_calls):
    
    retry_patience = 3
    
    for symbol in symbols:
        curr_time = datetime.datetime.now()
        details = []

        # Trust that the API knows what it's doing
                # Also leaving at default 5000 data points per call
        # (if it doesn't return 5000, that means we've reached the end)
        
        for _ in range(num_calls):
            retries = 0
            while retries <= retry_patience:
                data = TwelveDataAPI(symbol=symbol, end_date=curr_time)

                if data.get("status") == "error":
                    msg = data.get("message", "")
                    if "run out of API credits" in msg:
                        retries += 1
                        if retries > retry_patience:
                            print(f"Rate limit hit for {symbol}. Max retries exceeded. Skipping batch.")
                            break
                        print(f"Rate limit hit for {symbol}. Waiting 60s... (retry {retries}/{retry_patience})")
                        time.sleep(60)
                        continue  # retry same call
                    else:
                        print(f"Error retrieving data for {symbol}: {msg}")
                        break  # non-rate-limit error: skip
                else:
                    break  # successful call

            # If non-rate-limit error caused break, skip this symbol/batch
            if data.get("status") == "error" and "run out of API credits" not in msg:
                break

            batch = data.get("values", [])
            if not batch or len(batch) < 5000:
                # print(f"Retrieved all data for {symbol}.")
                break

            details.extend(batch)
            curr_time = datetime.datetime.strptime(batch[-1]["datetime"], "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes=30)

        if details:
            with open(os.path.join(path, f"{symbol}.json"), "w") as f:
                json.dump(details, f, indent=4)



if __name__ == "__main__":
    # 5 companies from each of the 11 S&P sectors
    # 9 more "important" large S&P companies for broader market conditions
    # And 15 "smaller" companies for variance injection
    # And of course the S&P data

    # For a total of 80

    print("Collecting data...")

    with open("data/equities/companies.txt", "r") as f:
        companies = [line.rstrip("\n") for line in f if line[0] != "#" and line != "\n"]
    
    write_data(companies)

    print("Finished collecting data.")
    