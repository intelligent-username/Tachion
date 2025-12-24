"""
API Call Utils
"""

import os
import requests
import datetime

from dotenv import load_dotenv

load_dotenv()

def TwelveDataAPI(url="https://api.twelvedata.com/time_series",
                   interval="30min", outputsize=5000, format="JSON", end_date = datetime.datetime.now(), apikey=None, symbol=None):
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
