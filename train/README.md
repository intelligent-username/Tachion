# Train

This folder contains the training scripts & configs. Relies heavily on `core/`.

Each asset class has its own subfolder, and, within each subfolder, we train three models (one for the 2.5th percentile, one for the 50th, and one for the 97.5th), that way we get a 95% confidence interval for our predictions.

- Training scripts:
  - Import features from `core/features.py`
  - Align targets for the chosen prediction horizon
  - Fit models using XGBoost with quantile loss
  - Save trained artifacts in `models/`

All scripts are designed to be reusable across assets and quantiles.
