# Train

This folder contains the training scripts & configs. Relies heavily on `core/`.

We'll be training a DeepAR model on each asset class:

- Equities
- Crypto
- Forex
- Oil
- Gold
- Silver
- Interest rate cut/hike probabilities

Note that the predictions for the latter 4 won't be as strong, as they're most strongly affected by macroeconomic factors such as the current geopolitical climate, central bank policies, and supply chain issues, which aren't captured in historical price data alone. However, we can still get some signal from historical price movements.

Some notes:

- The DeepAR model will return a probability distribution. This is especially nice for making quantile-based predictions and going extra risky/conservative based on your risk tolerance. It will give lots of data for making a decision.
- Once the DeepAR model is trained, when making inferences, we don't actually need as long of a history window as we do for training. For example, we could just go with the past 10 days instead of the past 5 years. This'll speed up inferences significantly.
- The equities will have a few engineered features relating to the returns on the S&P500 indeex as an **covariate**/indicator/predictor/feature for broader market health. THIS IS A **PAST** COVARIATE.
- Similarly, the crypto models will have engineered features based on **Bitcoin** returns (studies often show a high Granger Causality between BitCoin and other cryptos, so it's a useful feature). THIS IS A **PAST** COVARIATE.
- The interest rate hike/cut probability models will have engineered features based on recent Fed announcements, recent CPI data, and recent unemployment data. These will take the most work to collect data on and cleanly engineer.

Each asset class has its own subfolder, and, within each subfolder, we train three models (one for the 2.5th percentile, one for the 50th, and one for the 97.5th), that way we get a 95% confidence interval for our predictions.

- Training scripts:
  - Import features from `core/features.py`
  - Align targets for the chosen prediction horizon
  - Fit interest rates / models using XGBoost with quantile loss
    - SMOTE to handle class imbalance (since usually we hold rates steady)
  - Save trained artifacts in `models/`

All scripts are designed to be reusable across assets and quantiles.
