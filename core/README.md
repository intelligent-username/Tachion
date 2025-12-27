# Core

The `Core` module is for any code that is repeated. It contains utilities for:

- API calls
- Splitting into features/label
- Initiating constants
- Configuring the model (defining its architecture fully)
- Enforcing time context windows, horizons, and quantiles.
- Anything else that's used across multiple modules

## What each file is for

- `tapi.py`     # For making API calls to TwelveData
- `bapi.py`      # For making API calls to Binance.
- `oapi.py`      # For making API calls to Oanda.
- `fapi.py`      # For making API calls to the Federal Reserve using the fredapi library.