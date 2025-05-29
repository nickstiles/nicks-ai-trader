import yfinance as yf

def fetch_close_prices(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d"
) -> list[float]:
    """
    Fetch historical close prices for a given ticker.

    Args:
      ticker: Stock symbol (e.g., "AAPL").
      period: Data period (e.g., "1mo", "3mo", "1y").
      interval: Data interval (e.g., "1d", "1h").

    Returns:
      List of closing prices as floats.
    """
    # Download data
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    # Extract close prices
    return df["Close"].dropna().tolist()