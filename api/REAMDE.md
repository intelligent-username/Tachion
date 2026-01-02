# API

This folder contains utilities for Tachion's PROPRIETARY API. It's all about the model's predictions, front end's interaction with it, and so forth. There shouldn't be too any calls to external APIs here, and there shouldn't be too many endpoints either, since really we're just serving predictions. When calling the API for new data collection, it's still done with this API, but the functionality is defined elsewhere since it's mostly referencing external APIs and running miscellaneous scripts.

## Api Design

### Predictions

POST /api/predict
```json
{
    "asset": "AAPL",
    "horizon": 7,
    "data": "CSV string"
}
```

Response:
```json
{
    // These are the predictions
    "timestamps": [...],
    "medians": [...],
    "lower_95s": [...],
    "upper_95s": [...],
    "metadata": {"model": "LGBM_quantile_v1"}
}
```
