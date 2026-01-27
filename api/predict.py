"""
API endpoints for Tachion frontend.

1) GET /api/history - Fetch historical data for a symbol
2) POST /api/predict - Run prediction for a symbol
"""

import os
import sys
import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.apis.yfapi import YFinanceAPI
from core.apis.biapi import BinanceAPI
from core.apis.oaapi import OandaAPI

router = APIRouter()


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


# Helper to get historical data based on asset class
def get_historical_data(symbol: str, asset_class: str):
    """Fetch historical data using the appropriate API."""
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)  # 1 year of data
    
    if asset_class == "equities":
        result = YFinanceAPI(symbol, start_date=start_date, end_date=end_date)
    elif asset_class == "crypto":
        # Binance API
        result = BinanceAPI(symbol, start_date=start_date, end_date=end_date)
    elif asset_class in ["forex", "comm"]:
        # OANDA API for forex and commodities
        result = OandaAPI(symbol, start_date=start_date, end_date=end_date)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown asset class: {asset_class}")
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "API error"))
    
    return result.get("values", [])


@router.get("/history")
async def get_history(
    symbol: str = Query(..., description="Symbol to fetch"),
    asset_class: str = Query(..., description="Asset class: equities, crypto, forex, comm")
):
    """Get historical price data for a symbol."""
    try:
        values = get_historical_data(symbol, asset_class)
        
        # Format for frontend
        data = [
            {"timestamp": v["datetime"], "value": v["close"]}
            for v in values
            if v.get("close") is not None
        ]
        
        return {"data": data}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        last_price = last_value["close"]
        last_date = datetime.datetime.strptime(last_value["datetime"], "%Y-%m-%d")
        
        # TODO: Load actual model from models/deepar_{asset_class}.pt
        # For now, generate mock predictions based on historical volatility
        
        # Calculate historical volatility
        closes = [v["close"] for v in values if v.get("close")]
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
