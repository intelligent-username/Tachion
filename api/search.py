from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SearchRequest(BaseModel):
    symbol: str
    type: str
    timespan: str = 'max'


@router.post('/search')
async def search(request: SearchRequest):
    print(f"Request received with Symbol {request.symbol}, asset type {request.type}, and timespan {request.timespan}")
    return {'status': 'ok'}
