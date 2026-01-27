# Core

The `Core` module is for any code that is repeated. It contains utilities for:

- API calls
- Data processing
- Splitting into features/label
- Initiating constants
- Configuring the model (defining its architecture fully)
- Enforcing time context windows, horizons, and quantiles.
- Anything else that's used across multiple modules

Below are some of the important files and folders in this project.

## Files

These files are wrappers for making API calls:

```md
apis/              # Wrappers for API calls
├── biapi.py       # Binance API (crypto)
├── frapi.py       # FRED API (interest rates)
├── oaapi.py       # OANDA API (forex, comm)
├── tdapi.py       # TwelveData API (equities)
└── yfapi.py       # yfinance API (for inference)
```

These files do math and other operations for the preprocessing/feature engineering step:

```md
processor/         # Math for preprocessing
├── dw.py          # Date features
├── lr.py          # Log returns, volume change
├── ma.py          # Moving average calculator
├── pw.py          # Progress for long processes
└── rv.py          # Rolling volatility

training/          # Training configurations
├── config.py      # Argument parsing config
├── constants.py   # Training constants
└── loads.py       # Loaders

```
