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

def call_specific_td(path, symbols, num_calls, outputsize=5000, rate_limit=8):
    """
    Make Specific Calls to the TwelveData API
    """

    calls_this_minute = 0
    minute_start = time.time()

    for symbol in symbols:
        curr_time = datetime.datetime.now()
        details = []

        file_path = os.path.join(path, f"{symbol}.json")
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

                while True:
                    # RATE LIMIT
                    if calls_this_minute >= rate_limit:
                        elapsed = time.time() - minute_start
                        if elapsed < 60:
                            sleep_time = 60 - elapsed + 2
                            print(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                        calls_this_minute = 0
                        minute_start = time.time()

                    calls_this_minute += 1
                    data = TwelveDataAPI(symbol=symbol, end_date=curr_time)

                    if data.get("status") == "error":
                        msg = data.get("message", "")
                        if "run out of API credits" in msg:
                            # RATE LIMIT
                            elapsed = time.time() - minute_start
                            sleep_time = max(60 - elapsed, 1) + 2
                            print(f"Rate limit hit mid-call. Sleeping {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                            calls_this_minute = 0
                            minute_start = time.time()
                            continue
                        else:
                            print(f"Error retrieving data for {symbol}: {msg}")
                            break

                    break  # successful call

                if data.get("status") == "error":
                    break

                batch = data.get("values", [])
                if not batch:
                    break

                details.extend(batch)

                if len(batch) < outputsize:
                    break

                curr_time = datetime.datetime.strptime(
                    batch[-1]["datetime"], "%Y-%m-%d %H:%M:%S"
                ) - datetime.timedelta(minutes=30)

            if details:
                with open(file_path, "w") as f:
                    json.dump(details[::-1], f, indent=4)

        else:
            with open(file_path, "r") as f:
                existing_data = json.load(f)

            fmt = "%Y-%m-%d %H:%M:%S"
            last_dt = datetime.datetime.strptime(existing_data[-1]["datetime"], fmt)
            new_data = []

            while True:

                while True:
                    if calls_this_minute >= rate_limit:
                        elapsed = time.time() - minute_start
                        if elapsed < 60:
                            sleep_time = 60 - elapsed + 2
                            print(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                        calls_this_minute = 0
                        minute_start = time.time()

                    calls_this_minute += 1
                    data = TwelveDataAPI(
                        symbol=symbol,
                        start_date=last_dt + datetime.timedelta(minutes=30)
                    )

                    if data.get("status") == "error":
                        msg = data.get("message", "")
                        if "run out of API credits" in msg:
                            elapsed = time.time() - minute_start
                            sleep_time = max(60 - elapsed, 1) + 2
                            print(f"Rate limit hit mid-call. Sleeping {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                            calls_this_minute = 0
                            minute_start = time.time()
                            continue
                        elif "No data is available" in msg:
                            print(f"No new data for {symbol}; already up to date.")
                            break
                        else:
                            print(f"Error fetching {symbol}: {msg}")
                            break

                    break

                if data.get("status") == "error":
                    break

                batch = data.get("values", [])
                if not batch:
                    break

                batch = [
                    row for row in batch
                    if datetime.datetime.strptime(row["datetime"], fmt) > last_dt
                ]

                if not batch:
                    break

                new_data.extend(batch)
                last_dt = datetime.datetime.strptime(batch[-1]["datetime"], fmt)

                if len(batch) < outputsize:
                    break

            if new_data:
                with open(file_path, "w") as f:
                    json.dump(existing_data + new_data, f, indent=4)


def TwelveDataAPI(url="https://api.twelvedata.com/time_series",
                   interval="30min", outputsize=5000, format="JSON",
                   start_date=None, end_date=None, apikey=None, symbol=None):
    
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
    if end_date is None:
        end_date = datetime.datetime.now()

    if apikey is None:
        apikey = os.getenv("TD_KEY")

    if not symbol:
        raise ValueError("Symbol can't be blank. Please provide a valid symbol.")

    if format.upper() not in ["JSON", "CSV"]:
        raise ValueError("format must be 'JSON' or 'CSV'")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": format
    }
    if start_date:
        params["start_date"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
    if end_date:
        params["end_date"] = end_date.strftime("%Y-%m-%d %H:%M:%S")

    headers = {
        "Authorization": f"apikey {apikey}",
        "User-Agent": "tachion-data-collector",
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    if format.upper() == "CSV":
        return response.text
    else:
        return response.json()



