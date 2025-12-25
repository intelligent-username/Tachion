"""
Twelve Data API Call Utils
"""

import os
import requests
import datetime
import json
import time

from dotenv import load_dotenv

load_dotenv()

def call_specific(path, symbols, num_calls):
    """
    Make Specific Calls to the TwelveData API 
 
    """

    # JUST REALIZED there's a TwelveData library and I didn't have to make it this complex.
    # Bruh
    
    retry_patience = 3
    
    for symbol in symbols:
        curr_time = datetime.datetime.now()
        details = []

        # Trust that the API knows what it's doing
                # Also leaving at default 5000 data points per call
        # (if it doesn't return 5000, that means we've reached the end)

        file_path = os.path.join(path, f"{symbol}.json")

        # If the file path exists, see what the latest date in there is, and iterate forward from there to today to append.

        is_fresh = True

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if data:
                    latest_update_date = datetime.datetime.strptime(
                        data[-1]["datetime"], "%Y-%m-%d %H:%M:%S"
                    )
                    print(f"Found existing data for {symbol} up to {latest_update_date}. Updating...")
                    is_fresh = False

            except Exception as e:
                print(f"Could not parse existing file for {symbol}. Treating as fresh. Error: {e}")
                os.remove(file_path)
                is_fresh = True

        if is_fresh:
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
                            print(f"Rate limit hit for {symbol}. Waiting 58s... (retry {retries}/{retry_patience})")
                            time.sleep(58)
                            continue  # retry same call
                        else:
                            print(f"Error retrieving data for {symbol}: {msg}")
                            break  # non-rate-limit error: skip
                    else:
                        break  # successful call

                # If non-rate-limit error caused break, skip this symbol/batch
                if data.get("status") == "error" and "run out of API credits" not in data.get("message", ""):
                    break

                batch = data.get("values", [])
                if not batch or len(batch) < 5000:
                    # print(f"Retrieved all data for {symbol}.")
                    break

                details.extend(batch)
                curr_time = datetime.datetime.strptime(batch[-1]["datetime"], "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes=30)

            if details:
                with open(file_path, "w") as f:
                    json.dump(details[::-1], f, indent=4)
        else:
            # Otherwise it's not fresh, so iterate from latest date until cur_time

            # Load existing data
            with open(file_path, "r") as f:
                existing_data = json.load(f)

            fmt = "%Y-%m-%d %H:%M:%S"
            last_dt = datetime.datetime.strptime(existing_data[-1]["datetime"], fmt)

            new_data = []

            while True:
                data = TwelveDataAPI(
                    symbol=symbol,
                    start_date=last_dt + datetime.timedelta(minutes=30)
                )

                if data.get("status") == "error":
                    print(f"Error fetching {symbol}: {data.get('message','')}")
                    break

                batch = data.get("values", [])
                if not batch:
                    break

                # Keep only rows after last saved datetime
                batch = [row for row in batch if datetime.datetime.strptime(row["datetime"], fmt) > last_dt]

                if not batch:
                    break

                new_data.extend(batch)
                last_dt = datetime.datetime.strptime(batch[-1]["datetime"], fmt)

                if len(batch) < 5000:
                    break

            # Append new data to existing and write
            if new_data:
                full_data = existing_data + new_data
                with open(file_path, "w") as f:
                    json.dump(full_data, f, indent=4)

def TwelveDataAPI(url="https://api.twelvedata.com/time_series",
                   interval="30min", outputsize=5000, format="JSON", start_date = None , end_date = datetime.datetime.now(), apikey=None, symbol=None):
    """  
    :param url: TwelveData's Rest API Endpoint
    :param interval: time interval. 
            Must be one of the following values: 
                1min, 5min, 15min, 30min, 1h, 4h, 1day, 1week, 1month
    :param outputsize: How many data points to return. Maximum is 5000.
    :param format: The format of the data returned.
            Must be one of the following:
                JSON, CSV
                Pandas DataFrames can be created via the SDK
    :param end_date: The end date for the data retrieval. Defaults to present date and time.
    :param apikey: secret key. Do not touch. Will use environment variable TD_KEY.
    :param symbol: The specific symbol to retrieve. MAKE SURE IT'S NOT BLANK.
            For equities, just input the symbol
            For Crypto, input the symbol followed by slash and the currency (e.g., BTC/USD)
            For FOREX, input the currency pair separated by slash (e.g., EUR/USD)
            For Gold, input XAU/USD
            For Silver, input XAG/USD
            For Brent Crude (International) Oil, input XBR/USD 
    
    :return: Response object from TwelveData API

    """
    
    if apikey is None:
        apikey = os.getenv("TD_KEY")
    
    if symbol is None:
        raise ValueError("Symbol can't be blank. Please provide a valid symbol.")
    
    if format.upper() not in ["JSON", "CSV"]:
        raise ValueError("format must be 'JSON' or 'CSV'")

    
    details = {
        "apikey": apikey,
        "symbol": symbol,
        "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S") if start_date else None,
        "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        "interval": interval,
        "outputsize": outputsize,
        "format": format
    }

    response = requests.get(url, params=details)

    response.raise_for_status()

    if format.upper() == "CSV":
        return response.text
    else:
        return response.json()


