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

```md
apis/              # Wrappers for API calls
├── biapi.py       # Binance API (crypto)
├── frapi.py       # FRED API (interest rates)
├── oaapi.py       # OANDA API (forex, commodities)
├── tdapi.py       # TwelveData API (equities)
└── yfapi.py       # yfinance API (equities inference)
```

```md
processor/         # Reusable math for preprocessing
├── dw.py          # Date features
├── lr.py          # Log returns, volume change
├── ma.py          # Moving average calculator
├── pw.py          # Progress Writer: prints ratio
└── rv.py          # Rolling volatility calculator
```
