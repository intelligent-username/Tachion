# Data

The data folder is for:

- fetching data from an API or local source.
- cleaning and preprocessing raw data.
- storing processed datasets for training and evaluation.
- Data is cached so as to avoid redundant API calls, re-processing, and to speed up experimentation.

Each dataset is saved in CSV format for easy access during training and evaluation. This data won't be pushed to GitHub right now, but can be easily fetched using the `collector` scripts. Note that you will need to get your own API keys for this. Simply place them in an `.env` file in the main directory of the project.

## Features

Since we will train multiple different models, each model will expect a (slightly) different set of features.

### For Stocks

Note that they are US-only for now (if not a US stock, the model will exclude the S&P feature, but in the future I need to integrate a more robust way of getting a market-wide indicator (country-by-country)). Also, it would be good to add sector-specific indices as features. These are decisions to make when adding final polishes to the project. Of course, for the most optimal micro-optimization, we can add things like earnings reports, and we can train a single model per stock, but that's way out of scope right now.

<!-- "item_id" in DeepAR terminology -->
- `Ticker`                    <!-- Stock ticker symbol -->

- `Timestamp`                 <!-- In yyyy-mm-dd-hh-mi-ss format -->

<!-- "target" in DeepAR terminology -->
- `Price`                     <!-- Stock price in dollars -->

- `log_return`                <!-- -log(today's price / yesterday's price) -->

- `Volume`                    <!-- Number of shares traded in thousands -->

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


## For Cryptocurrencies

## For ETFs

## For FOREX

## For Commodities

## Data Storage

Each asset class (stocks, crypto, ETFs, FOREX, commodities) will have its own subfolder within this `data/` directory.

Within each subfolder,

- Raw data will be stored in the `raw/` directory of this folder. These are stored in JSON and come straight from the data provider API.
- Processed datasets are stored in the `processed/` directory.
