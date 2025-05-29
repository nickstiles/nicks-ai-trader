import numpy as np

def run_backtest(
    prices: list[float],
    short_window: int = 5,
    long_window: int = 20,
    initial_capital: float = 100000.0,
    quantity: int = 10
) -> dict:
    """
    Run a simple moving-average crossover backtest.

    Returns a dict with portfolio_values, trade_log, total_return, num_trades, and win_rate.
    """
    # Generate signals
    signals = []
    for i in range(len(prices)):
        if i < long_window:
            signals.append('hold')
        else:
            short_ma = sum(prices[i-short_window+1:i+1]) / short_window
            long_ma = sum(prices[i-long_window+1:i+1]) / long_window
            if short_ma > long_ma:
                signals.append('buy')
            elif short_ma < long_ma:
                signals.append('sell')
            else:
                signals.append('hold')

    # Simulate portfolio
    cash = initial_capital
    holdings = 0
    portfolio_values = []
    trade_log = []

    for i, signal in enumerate(signals):
        price = prices[i]
        if signal == 'buy' and cash >= price * quantity:
            cash -= price * quantity
            holdings += quantity
            trade_log.append({'index': i, 'action': 'buy', 'price': price, 'quantity': quantity})
        elif signal == 'sell' and holdings >= quantity:
            cash += price * quantity
            holdings -= quantity
            trade_log.append({'index': i, 'action': 'sell', 'price': price, 'quantity': quantity})
        portfolio_values.append(cash + holdings * price)

    # Compute metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    num_trades = len(trade_log)
    # Compute win rate for full round-trip trades
    wins = 0
    sell_events = 0
    for j in range(1, len(trade_log)):
        prev, curr = trade_log[j-1], trade_log[j]
        if prev['action'] == 'buy' and curr['action'] == 'sell':
            sell_events += 1
            if curr['price'] > prev['price']:
                wins += 1
    win_rate = wins / sell_events if sell_events > 0 else 0.0

    return {
        'portfolio_values': portfolio_values,
        'trade_log': trade_log,
        'total_return': round(total_return, 4),
        'num_trades': num_trades,
        'win_rate': round(win_rate, 4),
    }