import os

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
USE_PAPER = True
MIN_PRED_RETURN = 0.36
TRADE_MANAGER_URL = os.getenv("TRADE_MANAGER_URL", "http://trade_manager:8002")

TOP_TICKERS = [
    "TSLA", "AAPL", "NVDA", "AMZN", "AMD",
    "META", "MSFT", "AMC", "GOOGL", "PLTR",
    "BAC", "BABA", "GOOG", "MARA", "INTC",
    "SOFI", "NIO", "COIN", "NFLX", "F",
    "RIVN", "DIS", "PYPL", "SNAP", "C"
]

FEATURE_COLS = [
    "delta", "moneyness", "tlt_return_5d", "delta_iv",
    "stoch_d", "stk_range_pct_zscore", "implied_volatility",
    "delta_x_moneyness_resid"
]