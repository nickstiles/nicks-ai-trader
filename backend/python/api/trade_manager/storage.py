from api.trade_manager.alpaca import submit_close_order, submit_order_to_alpaca, get_account_cash, get_order
from api.trade_manager.db import SessionLocal
from api.trade_manager.models import Trade, AccountSnapshot
from api.signal_generator.models import TradeSignal

import logging
import datetime as dt
import pandas as pd
import time

def get_open_trades():
    """
    Return a list of open trades with all relevant fields, including live stats.
    """
    db = SessionLocal()
    try:
        return db.query(Trade).filter(Trade.status == "open").all()
    finally:
        db.close()

def get_closed_trades():
    """
    Return a list of closed Trade ORM instances so Pydantic can
    automatically pick up all fields (including cash balances).
    """
    db = SessionLocal()
    try:
        return (
            db.query(Trade)
              .filter(Trade.status == "closed")
              .order_by(Trade.closed_at.desc())
              .all()
        )
    finally:
        db.close()


def get_total_open_trades() -> int:
    db = SessionLocal()
    try:
        return db.query(Trade).filter(Trade.status == "open").count()
    finally:
        db.close()

def get_open_trade_count_for_ticker(ticker: str) -> int:
    db = SessionLocal()
    try:
        return (
            db.query(Trade)
              .filter(Trade.status == "open", Trade.ticker == ticker)
              .count()
        )
    finally:
        db.close()

def is_duplicate(option_symbol: str) -> bool:
    db = SessionLocal()
    try:
        return (
            db.query(Trade)
              .filter(
                  Trade.option_symbol == option_symbol,
                  Trade.status == "open"
              )
              .first() is not None
        )
    finally:
        db.close()

def get_account_snapshots(limit: int = 100):
    db = SessionLocal()
    try:
        rows = db.query(AccountSnapshot).order_by(AccountSnapshot.ts.desc()).limit(limit).all()
        return [
            {
                "ts":   row.ts.isoformat(),
                "cash": row.cash,
                "positions_value": row.positions_value,
                "total_equity":    row.total_equity,
            }
            for row in rows
        ]
    finally:
        db.close()

def add_trade(signal, quantity: int):
    """
    Submit an order to Alpaca, then persist the trade only if Alpaca succeeds.
    """
    # 1) Submit to Alpaca
    try:
        order = submit_order_to_alpaca(
            signal.option_symbol,
            quantity
        )
    except Exception as e:
        logging.error(f"Failed to submit order to Alpaca: {e}")
        return {"status": "error", "reason": str(e)}
    
    cash_on_entry = get_account_cash()

    max_retries = 5
    for attempt in range(max_retries):
        order = get_order(order.id)
        if order.filled_avg_price is not None and order.filled_qty:
            break
        time.sleep(1)

    avg_price  = float(order.filled_avg_price) if order.filled_avg_price is not None else signal.option_price
    filled_qty = float(order.filled_qty) if order.filled_qty else quantity
    capital_used = avg_price * filled_qty * 100

    # 2) Persist to DB
    db = SessionLocal()
    try:
        trade = Trade(
            option_symbol         = signal.option_symbol,
            ticker                = signal.ticker,
            entry_price           = avg_price,
            quantity              = filled_qty,
            capital_used          = capital_used,
            placed_at             = dt.datetime.now(dt.timezone.utc),
            status                = "open",
            pred_return           = signal.pred_return,
            strike                = signal.strike,
            expiry                = signal.expiry,
            cash_balance_on_entry = cash_on_entry,
            alpaca_order_id       = order.id
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return {"status": "ok", "db_id": trade.id}
    except Exception as e:
        db.rollback()
        logging.error(f"Failed to persist trade to DB: {e}")
        raise
    finally:
        db.close()

def close_trade(option_symbol: str, reason: str):
    """
    Submit a close order to Alpaca, then update the DB only if Alpaca succeeds.
    """
    now = dt.datetime.now(dt.timezone.utc)

    # Enforce PDT rule
    if _count_intraday_closes(now) >= 3 and _is_same_day(option_symbol, now):
        logging.warning("PDT rule: 3 intraday round trips reached")
        return {"status": "rejected", "reason": "PDT limit reached"}

    # 1) Retrieve trade
    db = SessionLocal()
    try:
        trade = (
            db.query(Trade)
              .filter(
                  Trade.option_symbol == option_symbol,
                  Trade.status == "open"
              )
              .one_or_none()
        )
        if not trade:
            return {"status": "error", "reason": "Trade not found"}

        # 2) Submit close order
        try:
            close_order = submit_close_order(
                option_symbol,
                trade.quantity
            )
        except Exception as e:
            logging.error(f"Failed to submit close order to Alpaca: {e}")
            return {"status": "error", "reason": str(e)}
        
        cash_on_exit = get_account_cash()

        # 3) Compute PnL
        pnl_dollars = pnl_pct = None
        if close_order.filled_avg_price is not None and trade.entry_price:
            pnl_dollars = (close_order.filled_avg_price - trade.entry_price) * trade.quantity * 100
            pnl_pct     = (close_order.filled_avg_price - trade.entry_price) / trade.entry_price

        # 4) Update DB
        trade.status                = "closed"
        trade.closed_at             = now
        trade.close_reason          = reason
        trade.exit_price            = close_order.filled_avg_price
        trade.pnl_dollars           = pnl_dollars
        trade.pnl_pct               = pnl_pct
        trade.cash_balance_on_exit  = cash_on_exit
        trade.alpaca_close_order_id = close_order.id

        db.commit()
        return {"status": "ok"}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def _is_same_day(option_symbol: str, now: dt.datetime) -> bool:
    db = SessionLocal()
    try:
        trade = (
            db.query(Trade)
              .filter(Trade.option_symbol == option_symbol)
              .one_or_none()
        )
        return bool(trade and trade.placed_at.date() == now.date())
    finally:
        db.close()


def _count_intraday_closes(now: dt.datetime) -> int:
    """
    Count of closed trades that opened and closed today.
    """
    db = SessionLocal()
    try:
        week_ago = now - dt.timedelta(days=7)
        trades = (
            db.query(Trade)
              .filter(
                  Trade.closed_at != None,
                  Trade.placed_at != None,
                  Trade.status == "closed",
                  Trade.closed_at >= week_ago
              )
              .all()
        )
        return sum(
            1 for t in trades if t.placed_at.date() == t.closed_at.date()
        )
    finally:
        db.close()