# Core

The `Core` module is for any code that is repeated. It contains utilities for:

- API calls
- Splitting into features/label
- Initiating constants
- Configuring the model (defining its architecture fully)
- Enforcing time context windows, horizons, and quantiles.
- Anything else that's used across multiple modules

## What each file is for

```md
- `tdapi.py` - Contains TwelveDataAPI function and call_specific function to make API calls to Twelve Data.

```
