# Core

The `Core` module is for any code that is repeated. It contains utilities for:

- API calls
- Splitting into features/label
- Initiating constants
- Configuring the model (defining its architecture fully)
- Enforcing time context windows, horizons, and quantiles.
- Anything else that's used across multiple modules

## Files

```
apis/              # Wrappers for API calls
├──biapi.py        # To Binance.
├──frapi.py        # To FRED
├──oaapi.py        # To OANDA.
├──tdapi.py        # To TwelveData.
└──yfapi.py        # To the yfinance API.
```

```
processors/        # Repeatable math for preprocessing
├──ma.py           # Moving Average finder
├──rv.py           # Rolling Volatility calculator           
├──lr.py           # Log Returns calculator
└──dw.py           # Date writer (known covariates)
```
