# Tachion

<img src="imgs/logo.svg" width="128" height="128" alt="Logo">

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
   - [Using Conda](#using-conda)  
   - [Using Python / pip](#using-python--pip)  
3. [Usage](#usage)  
4. [Contributing](#contributing)  
5. [License](#license)  

---

Tachion is a tool for predicting financial returns using historical market data. Currently, it supports:

- Stocks  
- Crypto  
- Forex  
- Commodities (gold, silver, oil)  
- ETFs  
- Interest rate hike/cut probabilities  

Tachion provides point predictions and approximate 95% confidence intervals, enabling users to decide whether or not it's a good time to buy or sell.

## Features

- Predict next-period returns or price levels for multiple asset classes  
- Quantile-based predictions to estimate confidence intervals  
- Supports CSV ingestion for historical data  
- Lightweight, extensible Python backend  
- Interactive visualization for historical and predicted values  

## Installation

The chief way to use Tachion is to go to [tachion.varak.dev](https://tachion.varak.dev) and use the hosted version. If you want to run it locally, follow the instructions below.

### Using Conda

```bash
conda create -n tachion python=3.11
conda activate tachion
conda install pip
pip install -r requirements.txt
````

### Using Python / pip

```bash
python3 -m venv tachion-env
source tachion-env/bin/activate  # macOS/Linux
tachion-env\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage

1. Prepare your CSV containing historical data (OHLCV or relevant features).
2. Start the backend server:

```bash
uvicorn api.main:app --reload
```

3. Send a POST request to `/predict` with your CSV and desired horizon.
4. Visualize predictions and confidence intervals through the integrated frontend.

## File Structure

```md
# Will be filled in once the project is finalized :)
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Ensure code adheres to PEP8 and includes tests where applicable.
3. Submit a pull request with a clear description of changes.


## License

This project is for non-commercial use only. Attribution is required. For commercial inquiries, please contact [inquiries@varak.dev](mailto:inquiries@varak.dev). For full license details, see the [LICENSE](LICENSE) file.
