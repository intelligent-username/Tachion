# Tachion

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-Non--Commercial-red)

![Logo](../imgs/Logo.svg)

## Table of Contents

1. [Features](#features)
2. [Prediction Mechanism](#prediction-mechanism)
3. [Usage](#usage)
   - [Installation](#installation)
   - [Dependencies](#dependencies)
   - [Environment Setup](#environment-setup)
   - [Data Collection](#data-collection)
4. [File Structure](#file-structure)
5. [Attributions](#attributions)
6. [Contributing](#contributing)
7. [License](#license)  

---

Tachion is a tool for forecasting market behaviour using historical market data. Currently, it supports:

- Stocks  
- Crypto  
- Forex  
- Commodities (gold, silver, oil)  
- Interest rate hike/cut probabilities  

Tachion provides point predictions and approximate 95% confidence intervals, enabling users to decide whether or not it's a good time to buy or sell.

## Features

- Predict next-period returns or price levels for multiple asset classes  
- Quantile-based predictions to estimate confidence intervals  
- Supports CSV ingestion for historical data  
- Lightweight, extensible Python backend  
- Interactive visualization for historical and predicted values  

## Prediction Mechanism

A different model will be trained for each class. For equities, cryptocurrencies, and commodities, we will use DeepAR (an RNN-based model) to predict the next-period log return based on historical OHLCV data and engineered features. For FOREX, we will use a similar architecture but with more macroeconomic features engineered in. For interest rate hike/cut probabilities, XGBoost will be used to classify the likelihood of a rate change based on historical data and economic indicators.

### DeepAR Explained

DeepAR is an autoregressive recurrent neural network architecture developed by Amazon for probabilistic time series forecasting. It learns patterns across multiple related time series simultaneously, generating not just point predictions but full probability distributions.

Once the learning is done, DeepAR would have generated a statistical distribution that best fits the simulations. We pick the statistical distribution to fit to beforehand, in this case, student's t-distribution, which is well-suited for financial returns due to its heavy tails. During inference, we just do some math based on the parameters of this distribution to get the desired quantiles (2.5th, 50th, 97.5th percentiles). Note, if the distribution doesn't fit well, we can always change it. If the math is "hard" to do, we can always just sample instead (or maybe even not even make a distribution but just plot a histogram directly and get quantiles from that).

This allows Tachion to provide confidence intervals alongside predictions, giving users a sense of uncertainty in the forecast.

See [this repository](https://github.com/intelligent-username/RNN) for more on RNNs and their implementation.

### XGBoost Explained

When it comes to interest rate predictions, we're no longer predicting continuous values, but categorical ones (hike/cut/hold). This requires us to perform classification instad of regression. There are many algorithms for classification. For this project, we'll go with XGBoost.

XGBoost (Extreme Gradient Boosting) is a highly efficient gradient boosting framework that excels at classification tasks. For interest rate predictions, it leverages economic indicators and historical patterns to classify the probability of rate hikes, cuts, or holds. Its ensemble of decision trees captures complex non-linear relationships in macroeconomic data. This, in turn, allows for decently robust predictions (as the Fed, in theory, is supposed to make data-driven decisions). However, it will require a lot of much feature engineering to get right.

See [this writeup](https://github.com/intelligent-username/Classification) for more on classification algorithms and XGBoost.

## Usage

The chief way to use Tachion is to go to [tachion.varak.dev](https://tachion.varak.dev) and use the hosted version.

However, if you want to run it locally, follow the instructions below.

### Installation

First, clone the repository:

```bash
git clone https://github.com/intelligent-username/tachion.git
```

### Dependencies

#### Using Conda

```bash
conda create -n tachion python=3.11 -y  # the -y auto-approves prompts
conda activate tachion
conda install pip
pip install -r requirements.txt
```

#### Using Python / pip

```bash
python -m venv tach
source tach/bin/activate  # macOS/Linux
tach\Scripts\activate     # Windows
pip install -r requirements.txt
# Make the project into a package
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root with the following API keys:

```bash
# TwelveData API (for equities)
TD_KEY=your_twelvedata_api_key

# OANDA API (for forex)
OANDA_KEY=your_oanda_api_token
OANDA_ACCOUNT_ID=your_oanda_account_id
```

- Get a TwelveData API key at [twelvedata.com](https://twelvedata.com)
- Get OANDA credentials at [oanda.com](https://www.oanda.com) (practice account works)
- Binance API (for crypto) requires no authentication

#### Data Collection

(Optional, could just run the models and not collect data to re-train them)

Collect historical market data using the provided collectors:

```bash
# Collect equity data (~5 years, 30-min intervals)
python -m data.equities.collector

# Collect cryptocurrency data (~5 years, 30-min intervals)
python -m data.crypto.collector

# Collect forex data (~10 years, 30-min intervals)
python -m data.forex.collector
```

Data is saved as JSON files in `data/{asset_class}/raw/`.

#### Testing

(Optional, if you clone this repo you can assume everything works as expected).

To run the test suite, navigate to the project root and execute:

```bash
pytest
```

#### Backend

To run the backend, execute:

```bash
uvicorn api.main:app --reload
```

3.Send a POST request to `/predict` with your data and desired horizon.

4.Visualize predictions and prediction intervals through the integrated frontend.

If tweaking any of the internals of the project, make sure to first read the provided `.md` (markdown) files in the folder you're working in to understand the corresponding specifications.

For a better understanding of how the project is structured, have a look at the file structure below.

Once the frontend is finalized, instructions for the frontend will be added as well

## File Structure

Below are some of the important files and folders in this project:

```bash
Tachion/
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env                    # Make this & put your API keys here
├── api/                    # FastAPI backend
│
├── core/
│   ├── README.md
│   ├── apis/               # API wrappers
│   │   ├── biapi.py        # For the Binance API
│   │   ├── frapi.py        # FRED API
│   │   ├── oaapi.py        # OANDA API
│   │   ├── tdapi.py        # TwelveData API
│   │   └── yfapi.py        # yfinance API (Yahoo Finance)
│   ├── processor/          # Data processing utilities
│   └── training/           # Training configs (constants, loaders)
|
├── data/                   # Data collection & engineer
│   ├── comm/               # Commodities (gold, silver, oil)
│   ├── crypto/
│   ├── equities/
│   ├── forex/              # Currencies
│   └── interest/           # Interest rate iddicators
|
├── frontend/
│   ├── components/         # React components
│   └── js/                 # Logic hooks
│
├── test/                   # Testing utils (core)
│
└── train/                  # Model definitions & training loop
```

## Attributions

Tachion uses the APIs from the following for market data:

|    Asset Class      |                   API                  |            Notes         |
| ------------------- |----------------------------------------|------------------------- |
| Equities            | [TwelveData](https://twelvedata.com)   |    Requires API key      |
| Crypto              | [Binance](https://binance.com)         | No authentication needed |
| Forex, Commodities  | [OANDA](https://oanda.com)             | API key requires account |
| Interest-Related    | [FRED](https://fred.stlouisfed.org//)  | Requires API key         |
| yfinance            | [yfinance](https://finance.yahoo.com/) | No authentication needed |

Other sources:

- Cleveland NowCast for real interest rates and CPI consensus:
  - [Cleveland Fed NowCast](https://www.clevelandfed.org/indicators-and-data/inflation-expectations)
  - [Original Excel File](https://www.clevelandfed.org/-/media/files/webcharts/inflationexpectations/inflation-expectations.xlsx?sc_lang=en&hash=C27818913D96CEDD80E3136B9946CFA7)

The original DeepAR paper (actually accessibly written):

Salinas et. al, "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", 22 Feb. 2019, [https://arxiv.org/abs/1704.04110](https://arxiv.org/abs/1704.04110), last accessed January 2026.

GluonTS, Amazon's official DeepAR library (implementation), which I took lots of inspiration from:

[https://github.com/awslabs/gluonts](https://github.com/awslabs/gluonts).

Documentation: [can be found here](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html).

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Ensure code adheres to PEP8 and includes tests where applicable.
3. Submit a pull request with a clear description of changes.

## License

Attribution is required. This project is for non-commercial use only. For commercial inquiries, please contact [inquiries@varak.dev](mailto:inquiries@varak.dev).

For full license details, see the [LICENSE](LICENSE).
