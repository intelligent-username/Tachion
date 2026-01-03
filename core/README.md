# Core

The `Core` module is for any code that is repeated. It contains utilities for:

- API calls
- Data processing 
- Splitting into features/label
- Initiating constants
- Configuring the model (defining its architecture fully)
- Enforcing time context windows, horizons, and quantiles.
- Anything else that's used across multiple modules

## Files

```
apis/              # Wrappers for API calls
├── biapi.py       # Binance API (crypto)
├── frapi.py       # FRED API (interest rates)
├── oaapi.py       # OANDA API (forex, commodities)
├── tdapi.py       # TwelveData API (equities)
└── yfapi.py       # yfinance API (equities inference)
```

```
processor/         # Reusable math for preprocessing
├── dw.py          # Date features
├── lr.py          # Log returns, volume change
├── ma.py          # Moving average calculator
├── pw.py          # Progress writer for printing completion ratio
└── rv.py          # Rolling volatility calculator
```
