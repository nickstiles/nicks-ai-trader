from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from data.ingestion import fetch_close_prices
from models.predict import predict_signal
from models.backtest import run_backtest

app = FastAPI(title="AI Trading Service", description="ML-based prediction API", version="0.1")

class PredictionRequest(BaseModel):
    ticker: str
    timeframe: Literal['1d', '1h', '5m']
    recent_prices: list[float]

class PredictionResponse(BaseModel):
    signal: Literal['buy', 'sell', 'hold']
    confidence: float

class BacktestRequest(BaseModel):
    ticker: str
    timeframe: Literal['1d', '1h', '5m']
    period: str = '1mo'
    short_window: int = 5
    long_window: int = 20
    quantity: int = 10

class BacktestResponse(BaseModel):
    portfolio_values: list[float]
    trade_log: list[dict]
    total_return: float
    num_trades: int
    win_rate: float

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

@app.post('/backtest', response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    # Fetch live or historical data
    prices = fetch_close_prices(req.ticker, period=req.period, interval=req.timeframe)
    # Run backtest
    results = run_backtest(
        prices,
        short_window=req.short_window,
        long_window=req.long_window,
        initial_capital=100000.0,
        quantity=req.quantity
    )
    return results