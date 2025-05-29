def moving_average(prices, window):
    """
    Compute the simple moving average for the last `window` prices.
    Returns None if not enough data is available.
    """
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def predict_signal(prices: list[float], short_window: int = 5, long_window: int = 20) -> dict:
    """
    Moving Average Crossover strategy:
      - 'buy' if short-term MA > long-term MA
      - 'sell' if short-term MA < long-term MA
      - 'hold' otherwise or if insufficient data
    Confidence is normalized difference between MAs.
    """
    # Insufficient data
    if len(prices) < long_window:
        return {"signal": "hold", "confidence": 0.0}

    short_ma = moving_average(prices, short_window)
    long_ma = moving_average(prices, long_window)
    if short_ma is None or long_ma is None:
        return {"signal": "hold", "confidence": 0.0}

    if short_ma > long_ma:
        signal = "buy"
    elif short_ma < long_ma:
        signal = "sell"
    else:
        signal = "hold"

    # Confidence based on MA divergence
    confidence = round(min(abs(short_ma - long_ma) / long_ma, 1.0), 2) if long_ma != 0 else 0.0
    return {"signal": signal, "confidence": confidence}