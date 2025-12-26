# Tachion

<img src="imgs/Logo.svg" width="256" height="256" alt="Logo">

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
   - [Using Conda](#using-conda)  
   - [Using Python / pip](#using-python--pip)  
3. [Usage](#usage)  
4. [File Structure](#file-structure)
5. [Contributing](#contributing)  
6. [License](#license)  

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

### XGBoost Explained

## Installation

The chief way to use Tachion is to go to [tachion.varak.dev](https://tachion.varak.dev) and use the hosted version. If you want to run it locally, follow the instructions below.

### Using Conda

```bash
conda create -n tachion python=3.11
conda activate tachion
conda install pip
pip install -r requirements.txt
```

### Using Python / pip

```bash
python -m venv tach
source tach/bin/activate  # macOS/Linux
tach\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
# Make the project into a package
pip install -e .
```

## Usage

1. Prepare your CSV containing historical data (OHLCV or relevant features).
2. Start the backend server:

```bash
uvicorn api.main:app --reload
```

3. Send a POST request to `/predict` with your CSV and desired horizon.

4. Visualize predictions and prediction intervals through the integrated frontend.

If tweaking any of the internals of the project, make sure to first read the provided `.md` (markdown) files in the folder you're working in to understand the corresponding specifications.

## File Structure

```md
Tachion/
├── LICENSE
├── README.md
├── requirements.txt
├── api/
│   ├── REAMDE.md
│   └── predict.py
├── core/
│   └── README.md
├── data/
│   ├── README.md
│   └── collect.py
├── frontend/
│   ├── README.md
│   ├── index.html
│   ├── styles.css
│   ├── main.js
│   ├── components/
│   │   ├── footer.js
│   │   ├── graph.js
│   │   ├── header.js
│   │   └── sidebar.js
│   └── js/
│       ├── api.js
│       ├── events.js
│       ├── ui.js
│       └── visualizer.js
├── imgs/
├── models/
│   └── README.md
├── test 
│   └── README.md
└── train/
    └── README.md
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Ensure code adheres to PEP8 and includes tests where applicable.
3. Submit a pull request with a clear description of changes.

## License

This project is for non-commercial use only. Attribution is required. For commercial inquiries, please contact [inquiries@varak.dev](mailto:inquiries@varak.dev). For full license details, see the [LICENSE](LICENSE) file.
