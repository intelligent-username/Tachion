# Data

The data folder is for:

- fetching data from an API or local source.
- cleaning and preprocessing raw data.
- storing processed datasets for training and evaluation.
- Data is cached so as to avoid redundant API calls, re-processing, and to speed up experimentation.

Each dataset is saved in CSV format for easy access during training and evaluation. This data won't be pushed to GitHub right now, but can be easily fetched using the `collector` scripts. Note that you will need to get your own API keys for this. Simply place them in an `.env` file in the main directory of the project.

## Feature Engineering

We'll be training multiple different models, each to predict a different target. As a result, each asset type will expect a (slightly) different set of features.

Equities and FOREX have time gaps (markets close at 4pm ET on weekdays and at 22:00 UTC Friday respectively, as well as holidays, etc.), so they require extra engineered features (day of the week, time since last  open, and so on) for the model to understand time context.

### For Equities

Note that they are US-only for now (if not a US stock, the model will exclude the S&P feature, but in the future I need to integrate a more robust way of getting a market-wide indicator (country-by-country)). Also, it would be good to add sector-specific indices (ETFs) as dynamic (covariate) features. These are decisions to make when adding final polishes to the project. Of course, for the most optimal micro-optimization, we can add things like earnings reports, and we can train a single model per stock, but that's way out of scope right now.

<!-- "item_id" in DeepAR terminology -->
- `Ticker`                    <!-- Stock ticker symbol -->

- `Timestamp`                 <!-- In yyyy-mm-dd-hh-mi-ss format -->

<!-- "target" in DeepAR terminology -->
- `Price`                     <!-- Stock price in dollars -->

- `log_return`                <!-- -log(today's price / yesterday's price) -->

- `day_of_week`               <!-- 0 = Monday, 6 = Sunday -->

- `day_of_month`              <!-- 1-31 -->

- `quarter`                   <!-- 1-4 -->

- `5_day_MA`                  <!-- 5-day moving average -->

- `50_day_MA`                 <!-- 50-day moving average -->

- `200_day_MA`                <!-- 200-day moving average -->

- `rolling_volatility_5`      <!-- Rolling volatility over 5 days -->

- `rolling_volatility_50`     <!-- Rolling volatility over 50 days -->

- `rolling_volatility_200`    <!-- Rolling volatility over 200 days -->

- `S&P_log_return`            <!-- Log return of the S&P 500 index. -->
<!-- Would need another indicator for non-US stocks -->

- `VIX`                       <!-- Volatility Index -->

For the S&P log return, we need to delay it by ~1 week or so in order to avoid data leakage (we won't have future S&P data when making predictions for the future price of a given stock).


## For Cryptocurrencies

## For FOREX

## For Commodities

## Data Storage

Each asset class (stocks, crypto, ETFs, FOREX, commodities) will have its own subfolder within this `data/` directory.

Within each subfolder,

- Raw data will be stored in the `raw/` directory of this folder. These are stored in JSON and come straight from the data provider API.
- Processed datasets are stored in the `processed/` directory.


## Data Collection

For any of the models, if we ever want to re-train it, we'll re-run the relevant `collector.py` script in the relevant subfolder. This will re-fetch data from the APIs and re-process it into CSVs.
Also, the processed (feature-engineered) files will be converted to `.parquet` format for faster everything.

### Equities

- We call the TwelveData API to collect historical data
- For inference, we use YFinance (since it's pretty reliable and fast), which we didn't use for historical data since it only goes back a limited time.

### Crypto

- We call the Binance API to collect historical data
- For inference, we also use Binance API, since it's extremely good all-around.

### Forex

- We use the Oanda API to collect historical data and for inference. Currencies are a lot more volatile but dependent on macroeconomic factors, so we'll be collecting ~15 years of data for training instead of 5.
