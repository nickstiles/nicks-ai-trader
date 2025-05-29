from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from data.ingestion import fetch_close_prices
from models.predict import predict_signal

app = FastAPI(title="AI Trading Service", description="ML-based prediction API", version="0.1")

class PredictionRequest(BaseModel):
    ticker: str
    timeframe: Literal['1d', '1h', '5m']
    recent_prices: list[float]

class PredictionResponse(BaseModel):
    signal: Literal['buy', 'sell', 'hold']
    confidence: float

@app.get("/")
def read_root():
    return {"message": "AI trading service is up"}

@app.post("/predict", response_model=PredictionResponse)
def predict_trade(req: PredictionRequest):
    print("📨 [FastAPI] Received prediction request:", req.dict())
    # If no prices provided, fetch last month of closes
    prices = req.recent_prices or fetch_close_prices(
        req.ticker, period="1mo", interval=req.timeframe
    )
    print("Prices:", prices)
    # Use the moving average crossover logic
    result = predict_signal(prices)
    print("📤 [FastAPI] Sending prediction response:", result)
    return result