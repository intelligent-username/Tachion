# API

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
