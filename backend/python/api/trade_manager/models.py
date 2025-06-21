from sqlalchemy import Column, Integer, String, Float, DateTime, Date
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import datetime as dt

Base = declarative_base()

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    option_symbol = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    entry_price = Column(Float)
    quantity = Column(Integer)
    capital_used = Column(Float)
    placed_at = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))
    exit_price = Column(Float)
    closed_at = Column(DateTime)
    close_reason = Column(String)
    pnl_dollars = Column(Float)
    pnl_pct = Column(Float)
    current_price = Column(Float)
    last_evaluated = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    status = Column(String)  # 'open' or 'closed'
    alpaca_order_id = Column(String)
    alpaca_close_order_id = Column(String)
    pred_return = Column(Float)
    strike = Column(Float)
    expiry = Column(Date)
    cash_balance_on_entry = Column(Float, nullable=False)
    cash_balance_on_exit  = Column(Float)

class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    ts              = Column(DateTime, primary_key=True, default=dt.datetime.now(dt.timezone.utc))
    cash            = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    total_equity    = Column(Float, nullable=False)
    
class TradeResponse(BaseModel):
    db_id: int = Field(alias="id")
    option_symbol: str
    ticker: str
    entry_price: float
    strike: float
    expiry: dt.date
    pred_return: float
    capital_used: float
    quantity: int
    placed_at: dt.datetime
    status: str
    cash_balance_on_entry: float
    current_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    last_evaluated: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    closed_at: Optional[dt.datetime] = None
    close_reason: Optional[str] = None
    pnl_dollars: Optional[float] = None
    alpaca_order_id: Optional[str] = None
    alpaca_close_order_id: Optional[str] = None
    cash_balance_on_exit: Optional[float] = None

    # Pydantic v2 config: read ORM attributes
    model_config = ConfigDict(from_attributes=True)