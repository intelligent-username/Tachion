"""
API endpoints for Tachion frontend.

1) POST /api/predict - Run prediction for a symbol
"""

import sys
import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.processor.ip import get_historical_data

router = APIRouter()


def _parse_datetime(value: str) -> datetime.datetime:
    text = str(value)
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'

    try:
        dt = datetime.datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported datetime format: {value}")


class PredictRequest(BaseModel):
    symbol: str
    asset_class: str
    horizon: int = 7


class PredictResponse(BaseModel):
    timestamps: list
    medians: list
    lower_95s: list
    upper_95s: list
    metadata: dict


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run model prediction for a symbol."""
    try:
        # Get recent historical data for context
        values = get_historical_data(request.symbol, request.asset_class)
        
        if not values:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Get the last price
        last_value = values[-1]
        last_price = float(last_value["close"])
        last_date = _parse_datetime(last_value["datetime"])
        
        # TODO: Load actual model from models/deepar_{asset_class}.pt
        # For now, generate mock predictions based on historical volatility
        
        # Calculate historical volatility
        closes = [float(v["close"]) for v in values if v.get("close") is not None]
        if len(closes) > 1:
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        else:
            volatility = 0.02  # Default 2% volatility
        
        # Generate prediction timestamps and values
        timestamps = []
        medians = []
        lower_95s = []
        upper_95s = []
        
        current_price = last_price
        for i in range(1, request.horizon + 1):
            pred_date = last_date + datetime.timedelta(days=i)
            timestamps.append(pred_date.strftime("%Y-%m-%d"))
            
            # Simple random walk with drift (placeholder for actual model)
            # Median stays relatively flat with slight trend
            drift = 0.0001 * i
            median = current_price * (1 + drift)
            
            # 95% CI grows with sqrt of time
            ci_width = last_price * volatility * 1.96 * (i ** 0.5)
            
            medians.append(round(median, 4))
            lower_95s.append(round(median - ci_width, 4))
            upper_95s.append(round(median + ci_width, 4))
        
        return PredictResponse(
            timestamps=timestamps,
            medians=medians,
            lower_95s=lower_95s,
            upper_95s=upper_95s,
            metadata={"model": f"deepar_{request.asset_class}", "horizon": request.horizon}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
