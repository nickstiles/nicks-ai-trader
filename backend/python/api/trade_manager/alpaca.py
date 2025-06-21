from api.signal_generator.config import ALPACA_API_KEY, ALPACA_SECRET_KEY

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def get_account_equity():
    account = trading_client.get_account()
    return float(account.equity)

def get_account_cash():
    account = trading_client.get_account()
    return float(account.options_buying_power)

def get_all_positions():
    positions = trading_client.get_all_positions()
    return positions

def get_order(order_id: str):
    order = trading_client.get_order_by_id(order_id)
    return order

def submit_order_to_alpaca(symbol: str, qty: int, limit_price: float = None):
    order_data = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        type=OrderType.MARKET if limit_price is None else OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2) if limit_price else None
    )
    order = trading_client.submit_order(order_data)
    return order

def submit_close_order(symbol: str, qty: int, limit_price: float = None):
    order_data = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        type=OrderType.MARKET if limit_price is None else OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2) if limit_price else None
    )
    return trading_client.submit_order(order_data)