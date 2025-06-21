from api.signal_generator.models import TradeSignal
from api.trade_manager.models import Trade, AccountSnapshot
from api.trade_manager.config import (
    CAPITAL_PER_TRADE_PERCENT, 
    MAX_OPEN_TRADES, 
    MAX_OPEN_TRADES_PER_TICKER,
    MAX_HOLD_DAYS,
    PCT_STOP_LOSS,
    PCT_TAKE_PROFIT
)
from api.trade_manager.storage import (
    is_duplicate,
    get_total_open_trades,
    get_open_trade_count_for_ticker,
    add_trade,
    close_trade,
    get_closed_trades
)
from api.trade_manager.alpaca import get_account_cash, get_all_positions
from api.trade_manager.fetchers import get_option_snapshot
from api.trade_manager.db import SessionLocal
from api.trade_manager.events import publish_update
from api.utils import is_market_open

import datetime as dt
import logging
import pandas_market_calendars as mcal
from statistics import mean

def handle_trade_signal(signal: TradeSignal):
    if is_duplicate(signal.option_symbol):
        return {"status": "rejected", "reason": "duplicate option"}

    if get_total_open_trades() >= MAX_OPEN_TRADES:
        return {"status": "rejected", "reason": "max open trades reached"}

    if get_open_trade_count_for_ticker(signal.ticker) >= MAX_OPEN_TRADES_PER_TICKER:
        return {"status": "rejected", "reason": "too many trades on this ticker"}

    cash = get_account_cash()
    capital_per_trade = cash * CAPITAL_PER_TRADE_PERCENT
    quantity = int(capital_per_trade // (signal.option_price * 100))

    if quantity <= 0:
        return {"status": "rejected", "reason": "option price too high for allocation"}

    add_trade(signal, quantity)
    publish_update({"type": "signal", "signal": signal.option_symbol})
    return {
        "status": "accepted",
        "option_symbol": signal.option_symbol,
        "quantity": quantity
    }

def trading_days_between(start: dt.datetime, end: dt.datetime) -> int:
    """
    Count the number of actual NYSE trading days >= start.date()
    and < end.date(), excluding weekends and market holidays.
    """
    if start >= end:
        return 0

    # Determine the calendar window: from start.date() up to (but not including) end.date()
    start_date = start.date()
    end_date = (end - dt.timedelta(days=1)).date()

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    # Each index entry is a trading day
    return len(schedule)

def evaluate_trades():
    """Synchronously evaluate open trades, persist live marks & PnL, then close if needed."""
    now = dt.datetime.now(dt.timezone.utc)
    closed = []

    db = SessionLocal()
    try:
        # 1) load all open trades
        trades = db.query(Trade).filter(Trade.status == "open").all()
        logging.info(f"🔎 Evaluating {len(trades)} open trades")

        positions = get_all_positions()
        pos_map = {p.symbol: float(p.current_price) for p in positions}

        for trade in trades:
            # unpack directly from the ORM
            symbol      = trade.option_symbol
            ticker      = trade.ticker
            entry_price = trade.entry_price or 0.0
            placed_at   = trade.placed_at

            current_price = pos_map.get(symbol)
            if current_price is None:
                # no live position, fall back to your polygon fetch
                polygon_sym = f"O:{symbol}"
                current_price = get_option_snapshot(ticker, polygon_sym)
                if current_price is None:
                    logging.warning(f"Could not fetch price for {symbol} (Pol.Symbol={polygon_sym})")
                    continue
            # 3) compute PnL
            pnl_dollars = (current_price - entry_price) * trade.quantity * 100
            pnl_pct     = ((current_price - entry_price) / entry_price) if entry_price else 0.0

            # 4) persist back onto the ORM object
            trade.current_price  = current_price
            trade.pnl_dollars    = pnl_dollars
            trade.pnl_pct        = pnl_pct
            trade.last_evaluated = now
            db.add(trade)

            if is_market_open():
                # 5) age-based close
                age_days = trading_days_between(placed_at, now)
                if age_days >= MAX_HOLD_DAYS:
                    close_trade(symbol, "time expiry")
                    closed.append(symbol)
                    continue

                # 6) return-based close
                if pnl_pct <= PCT_STOP_LOSS:
                    close_trade(symbol, "stop loss")
                    closed.append(symbol)
                elif pnl_pct >= PCT_TAKE_PROFIT:
                    close_trade(symbol, "take profit")
                    closed.append(symbol)

        # 7) commit all updates at once
        db.commit()

    except Exception:
        db.rollback()
        logging.exception("Error evaluating trades")
        raise
    finally:
        db.close()

    # notify front-ends via SSE
    publish_update({"type": "evaluate", "closed": closed})
    return {"closed_trades": closed}

def compute_metrics():
    trades = get_closed_trades()
    if not trades:
        return {"message": "No trades closed yet."}

    total = len(trades)
    wins = [t for t in trades if t.get("pnl_dollars", 0) > 0]
    losses = [t for t in trades if t.get("pnl_dollars", 0) <= 0]
    win_rate = len(wins) / total if total > 0 else 0

    avg_pnl = mean(t["pnl_dollars"] for t in trades if t.get("pnl_dollars") is not None)
    avg_return = mean(t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None)

    profit = sum(t["pnl_dollars"] for t in wins if t.get("pnl_dollars") is not None)
    loss = abs(sum(t["pnl_dollars"] for t in losses if t.get("pnl_dollars") is not None))
    profit_factor = round(profit / loss, 2) if loss > 0 else "∞"

    return {
        "total_trades": total,
        "win_rate": round(win_rate * 100, 2),
        "avg_pnl_dollars": round(avg_pnl, 2),
        "avg_return_pct": round(avg_return * 100, 2),
        "profit_factor": profit_factor
    }

def snapshot_account():
    """
    Fetch current Alpaca account + positions, compute total equity,
    and write a snapshot row to the DB.
    """
    # 1) get cash
    cash = get_account_cash()

    # 2) get positions and sum their market values
    positions = get_all_positions()
    positions_value = sum(
        float(p.qty) * float(p.current_price)
        for p in positions
    )

    # 3) total
    total_equity = cash + positions_value

    # 4) persist
    db = SessionLocal()
    try:
        snap = AccountSnapshot(
            ts=dt.datetime.now(dt.timezone.utc),
            cash=cash,
            positions_value=positions_value,
            total_equity=total_equity,
        )
        db.add(snap)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()