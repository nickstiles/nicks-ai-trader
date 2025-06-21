#!/usr/bin/env python3
"""
data_ingest.py

Nightly ingestion job that:
  1. Downloads stock OHLCV for a universe of tickers (e.g. S&P 500).
  2. Computes stock-level features (RSI, SMA, returns, etc.).
  3. Downloads option greeks/IV/skew/volume from Polygon.
  4. Computes option-level features (moneyness, DTE).
  5. Downloads VIX (via yfinance).
  6. Inserts/UPSERTs everything into Postgres:
      - daily_stock_features
      - daily_option_features
      - vix_data
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# PostgreSQL connection parameters
DB_USER = "myuser"
DB_PASSWORD = "N15s1331%"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "options_trading"
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Polygon API
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_API_KEY:
    print("ERROR: Please set your POLYGON_API_KEY environment variable.")
    sys.exit(1)

# Universe of tickers (e.g. S&P 500). We fetch from Wikipedia each run.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Date for which to ingest data (defaults to yesterday)
DATA_DATE = (datetime.now() - timedelta(days=1)).date().isoformat()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list:
    """
    Scrape the current S&P 500 tickers from Wikipedia.
    Returns a list of ticker strings.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    try:
        tables = pd.read_html(SP500_WIKI_URL)
    except Exception as e:
        logger.error(f"Failed to fetch tickers from Wikipedia: {e}")
        sys.exit(1)

    # On the SP500 page, the first table is the list of constituents
    df_sp500 = tables[0]
    tickers = df_sp500["Symbol"].tolist()
    # Some tickers have dots (e.g. BRK.B); yfinance expects '-' instead of '.'
    tickers = [t.replace(".", "-") for t in tickers]
    logger.info(f"Found {len(tickers)} tickers.")
    return tickers

def compute_stock_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of OHLCV for multiple tickers (indexed by ticker, date),
    compute stock-level features:
      - rsi_14
      - sma_5, sma_10
      - stk_return_1d
      - stk_vol_chg_pct
      - stk_range_pct
      - realized_vol_10d (using ATR as proxy)
      - price_change_3d
      - ma5_ma10_diff
      - atr_5
    Expects df_ohlcv columns: ['Date', 'Open','High','Low','Close','Volume','Ticker']
    Returns a DataFrame with at least:
      ['ticker','trade_date','stk_open','stk_high','stk_low','stk_close','stk_volume',
       'rsi_14','sma_5','sma_10','stk_return_1d','stk_vol_chg_pct','stk_range_pct',
       'realized_vol_10d','price_change_3d','ma5_ma10_diff','atr_5']
    """
    out_rows = []
    # Process per ticker
    for ticker, df_tk in df_ohlcv.groupby("Ticker"):
        df_tk = df_tk.sort_values("Date").reset_index(drop=True)
        df_tk["trade_date"] = df_tk["Date"].dt.date.astype(str)
        df_tk.rename(
            columns={
                "Open": "stk_open",
                "High": "stk_high",
                "Low": "stk_low",
                "Close": "stk_close",
                "Volume": "stk_volume"
            },
            inplace=True
        )

        # RSI(14)
        try:
            rsi = RSIIndicator(close=df_tk["stk_close"], window=14).rsi()
        except Exception:
            rsi = pd.Series(np.nan, index=df_tk.index)
        df_tk["rsi_14"] = rsi

        # SMA(5), SMA(10)
        df_tk["sma_5"]  = SMAIndicator(close=df_tk["stk_close"], window=5).sma_indicator()
        df_tk["sma_10"] = SMAIndicator(close=df_tk["stk_close"], window=10).sma_indicator()

        # 1-day return
        df_tk["stk_return_1d"] = df_tk["stk_close"].pct_change(1)

        # Volume change pct (1d)
        df_tk["stk_vol_chg_pct"] = df_tk["stk_volume"].pct_change(1)

        # Range pct = (High - Low)/Close
        df_tk["stk_range_pct"] = (df_tk["stk_high"] - df_tk["stk_low"]) / df_tk["stk_close"]

        # ATR(14) as proxy for realized_vol_10d
        df_tk["atr_5"] = AverageTrueRange(
            high=df_tk["stk_high"],
            low=df_tk["stk_low"],
            close=df_tk["stk_close"],
            window=5
        ).average_true_range()

        # Realized vol 10d = rolling std of returns * sqrt(252)
        df_tk["realized_vol_10d"] = df_tk["stk_return_1d"].rolling(window=10).std() * np.sqrt(252)

        # Price change over 3 days
        df_tk["price_change_3d"] = df_tk["stk_close"].pct_change(3)

        # ma5_ma10_diff
        df_tk["ma5_ma10_diff"] = df_tk["sma_5"] - df_tk["sma_10"]

        # Keep only rows where date == DATA_DATE
        df_today = df_tk[df_tk["trade_date"] == DATA_DATE].copy()
        if df_today.empty:
            continue

        # Select columns
        cols = [
            "ticker",
            "trade_date",
            "stk_open", "stk_high", "stk_low", "stk_close", "stk_volume",
            "rsi_14", "sma_5", "sma_10", "stk_return_1d",
            "stk_vol_chg_pct", "stk_range_pct",
            "realized_vol_10d", "price_change_3d",
            "ma5_ma10_diff", "atr_5"
        ]
        df_today["ticker"] = ticker
        out_rows.append(df_today[cols])

    if not out_rows:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "stk_open", "stk_high", "stk_low", "stk_close", "stk_volume",
            "rsi_14", "sma_5", "sma_10", "stk_return_1d",
            "stk_vol_chg_pct", "stk_range_pct",
            "realized_vol_10d", "price_change_3d",
            "ma5_ma10_diff", "atr_5"
        ])

    return pd.concat(out_rows, ignore_index=True)


def fetch_stock_ohlcv(tickers: list, as_of_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV for all tickers for the past 30 trading days, then we pick out as_of_date.
    Uses yfinance because Polygon's free tier may be limited for daily. 
    Returns DataFrame with columns: ['Date','Open','High','Low','Close','Volume','Ticker'].
    """
    start_dt = (datetime.fromisoformat(as_of_date) - timedelta(days=30)).date().isoformat()
    end_dt   = as_of_date

    logger.info(f"Downloading stock OHLCV from {start_dt} to {end_dt} for {len(tickers)} tickers...")
    all_data = []
    batch_size = 50  # yfinance can fetch multiple tickers at once
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = yf.download(
            tickers=" ".join(batch),
            start=start_dt,
            end=end_dt,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        # yf returns a panel if multiple tickers, else a DataFrame
        if isinstance(data, pd.DataFrame) and len(batch) > 1:
            # Multi-index columns: first level = ticker, second = OHLCV
            for tk in batch:
                if tk not in data.columns.levels[0]:
                    continue
                df_tk = data[tk].reset_index().rename(columns={"Date": "Date"})
                df_tk["Ticker"] = tk
                all_data.append(df_tk)
        else:
            # Single ticker
            df_tk = data.reset_index().rename(columns={"Date": "Date"})
            df_tk["Ticker"] = batch[0]
            all_data.append(df_tk)

        time.sleep(1)  # avoid spamming yfinance

    if not all_data:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume","Ticker"])
    return pd.concat(all_data, ignore_index=True)


def fetch_vix(as_of_date: str) -> pd.DataFrame:
    """
    Fetch VIX OHLCV for as_of_date (via yfinance ticker '^VIX').
    Returns a DataFrame with one row: ['trade_date','vix_open','vix_high','vix_low','vix_close','vix_volume'].
    """
    vix_ticker = "^VIX"
    start_dt = (datetime.fromisoformat(as_of_date) - timedelta(days=5)).date().isoformat()
    end_dt   = (datetime.fromisoformat(as_of_date) + timedelta(days=1)).date().isoformat()

    logger.info(f"Downloading VIX OHLCV for {as_of_date}...")
    df = yf.download(
        tickers=vix_ticker,
        start=start_dt,
        end=end_dt,
        group_by=None,
        auto_adjust=False,
        threads=False,
        progress=False,
    ).reset_index()

    df["trade_date"] = df["Date"].dt.date.astype(str)
    df_today = df[df["trade_date"] == as_of_date].copy()
    if df_today.empty:
        logger.warning(f"No VIX data for {as_of_date}")
        return pd.DataFrame(columns=["trade_date","vix_open","vix_high","vix_low","vix_close","vix_volume"])

    # Select relevant columns
    return df_today[["trade_date", "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={
            "Open": "vix_open",
            "High": "vix_high",
            "Low": "vix_low",
            "Close": "vix_close",
            "Volume": "vix_volume"
        }
    )


def fetch_option_contracts(ticker: str, as_of_date: str) -> list:
    """
    Using Polygon's /v3/reference/options/contracts endpoint, fetch all option contract symbols
    for a given underlying ticker. We can optionally filter by strike range or expiry range if desired.
    For simplicity, we fetch all contracts and filter by expiry >= as_of_date.

    Returns a list of option contract symbols (strings).
    """
    url = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "limit": 1000,      # max per page
        "apiKey": POLYGON_API_KEY
    }
    contracts = []
    logger.debug(f"  Fetching option contracts for {ticker}...")
    while True:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error(f"Error fetching contracts for {ticker}: {r.status_code} {r.text}")
            return contracts
        data = r.json()
        for row in data.get("results", []):
            # Only study contracts whose expiration >= as_of_date
            exp = row.get("expiration_date")
            if exp and exp >= as_of_date:
                contracts.append(row["ticker"])
        # Pagination: next_url
        next_url = data.get("next_url")
        if not next_url:
            break
        # next_url already contains the apiKey
        url = next_url
        params = None
        time.sleep(0.2)
    logger.debug(f"    Found {len(contracts)} contracts for {ticker}")
    return contracts


def fetch_option_data_for_contract(contract_symbol: str, as_of_date: str) -> dict:
    """
    Fetch a single day’s option snapshot (greeks, IV, OI, volume) for a contract:
    We use the /v3/snapshot/options/{contract_symbol} endpoint.

    Returns a dict with fields:
      {
        "underlying_symbol": "AAPL",
        "trade_date": "2025-06-05",
        "option_symbol": "...",
        "strike_price": float,
        "expiration_date": "2025-07-18",
        "option_type": "call"/"put",
        "implied_volatility": float,
        "delta": float,
        "gamma": float,
        "theta": float,
        "vega": float,
        "open_interest": int,
        "opt_volume": int,
        "ask": float,
        "bid": float,
        "underlying_price": float
      }
    If any field is missing, returns an empty dict.
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{contract_symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        logger.debug(f"    Failed snapshot for {contract_symbol}: {r.status_code}")
        return {}

    d = r.json().get("results", {})
    if not d or d.get("day") is None:
        return {}

    day = d["day"]
    greeks = d.get("greeks", {})
    return {
        "underlying_symbol": d.get("underlying_ticker"),
        "trade_date": as_of_date,
        "option_symbol": contract_symbol,
        "strike_price": float(d.get("strike_price", 0)),
        "expiration_date": d.get("expiration_date"),
        "option_type": d.get("contract_type"),  # "call" or "put"
        "implied_volatility": greeks.get("iv", np.nan),
        "delta": greeks.get("delta", np.nan),
        "gamma": greeks.get("gamma", np.nan),
        "theta": greeks.get("theta", np.nan),
        "vega": greeks.get("vega", np.nan),
        "open_interest": day.get("open_interest", 0),
        "opt_volume": day.get("volume", 0),
        "ask": day.get("ask", np.nan),
        "bid": day.get("bid", np.nan),
        "underlying_price": d.get("underlying_price", np.nan)
    }


def compute_option_features(df_opts: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of raw option snapshots (one row per contract):
    - Compute dte_days = (expiration_date - trade_date)
    - Compute moneyness = underlying_price / strike_price (float)
    - delta_gamma, delta_iv as placeholders (you may add additional derived features)
    Returns a DataFrame with columns matching daily_option_features:
      ['ticker','trade_date','implied_volatility','delta','gamma','theta',
       'iv_skew_atm','skew_vol','dte_days','moneyness','delta_gamma','delta_iv','vega','opt_volume']
    Note: We do NOT compute skew_vol or iv_skew_atm here, because that requires cross‐strike data;
    you could compute a per‐ticker skew by grouping by strike, but for simplicity we set them to NaN.
    """
    if df_opts.empty:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "implied_volatility", "delta", "gamma", "theta",
            "iv_skew_atm", "skew_vol", "dte_days", "moneyness", "delta_gamma", "delta_iv", "vega", "opt_volume"
        ])

    # Convert dates
    df_opts["expiration_date"] = pd.to_datetime(df_opts["expiration_date"]).dt.date
    df_opts["trade_date_dt"] = pd.to_datetime(df_opts["trade_date"]).dt.date
    df_opts["dte_days"] = (df_opts["expiration_date"] - df_opts["trade_date_dt"]).dt.days
    df_opts["moneyness"] = df_opts["underlying_price"] / df_opts["strike_price"]

    # Placeholders for iv_skew_atm, skew_vol, delta_gamma, delta_iv
    df_opts["iv_skew_atm"] = np.nan
    df_opts["skew_vol"] = np.nan
    df_opts["delta_gamma"] = np.nan
    df_opts["delta_iv"] = np.nan

    # Rename underlying_ticker → ticker
    df_opts.rename(columns={"underlying_symbol": "ticker"}, inplace=True)

    # Select columns to write
    return df_opts[[
        "ticker",
        "trade_date",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "iv_skew_atm",
        "skew_vol",
        "dte_days",
        "moneyness",
        "delta_gamma",
        "delta_iv",
        "vega",
        "opt_volume"
    ]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Starting data ingestion for date = {DATA_DATE}")

    # 1) Load the list of S&P 500 tickers
    tickers = get_sp500_tickers()

    # 2) Fetch stock OHLCV for the past 30 days, then filter to DATA_DATE
    df_stock_raw = fetch_stock_ohlcv(tickers, DATA_DATE)
    if df_stock_raw.empty:
        logger.error(f"No stock data for {DATA_DATE}. Exiting.")
        return

    # 3) Compute stock features
    df_stock_feats = compute_stock_features(df_stock_raw)
    if df_stock_feats.empty:
        logger.warning(f"No stock features computed for {DATA_DATE}.")
    else:
        logger.info(f"Computed stock features for {len(df_stock_feats)} tickers.")

    # 4) Fetch VIX for DATA_DATE
    df_vix = fetch_vix(DATA_DATE)
    if df_vix.empty:
        logger.warning("No VIX row to insert.")
    else:
        logger.info("Fetched VIX data.")

    # 5) For each ticker, fetch all valid option contract symbols
    all_option_rows = []
    for tk in tickers:
        try:
            contracts = fetch_option_contracts(tk, DATA_DATE)
        except Exception as e:
            logger.error(f"Error fetching contracts for {tk}: {e}")
            contracts = []
        time.sleep(0.2)  # rate‐limit

        # 6) For each contract, fetch option snapshot for DATA_DATE
        for c in contracts:
            opt_row = fetch_option_data_for_contract(c, DATA_DATE)
            if opt_row:
                all_option_rows.append(opt_row)
            time.sleep(0.1)  # rate‐limit

    if not all_option_rows:
        logger.warning("No option snapshots fetched.")
        df_option_feats = pd.DataFrame(columns=[
            "ticker", "trade_date", "implied_volatility", "delta", "gamma", "theta",
            "iv_skew_atm", "skew_vol", "dte_days", "moneyness", "delta_gamma", "delta_iv",
            "vega", "opt_volume"
        ])
    else:
        df_opts_raw = pd.DataFrame(all_option_rows)
        logger.info(f"Fetched {len(df_opts_raw)} raw option snapshots.")
        df_option_feats = compute_option_features(df_opts_raw)
        logger.info(f"Computed option features for {len(df_option_feats)} contracts.")

    # ─────────────────────────────────────────────────────────────────────────
    # 7) CONNECT TO POSTGRES AND UPSERT INTO TABLES
    # ─────────────────────────────────────────────────────────────────────────
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        # a) Insert/UPSERT daily_stock_features
        if not df_stock_feats.empty:
            logger.info("Writing to daily_stock_features...")
            df_stock_feats.to_sql(
                "daily_stock_features",
                conn,
                if_exists="append",  # make sure duplicates won’t cause errors; you could use ON CONFLICT in a custom upsert
                index=False,
                method="multi"
            )
        # b) Insert/UPSERT vix_data
        if not df_vix.empty:
            logger.info("Writing to vix_data...")
            df_vix.to_sql(
                "vix_data",
                conn,
                if_exists="append",
                index=False
            )
        # c) Insert/UPSERT daily_option_features
        if not df_option_feats.empty:
            logger.info("Writing to daily_option_features...")
            df_option_feats.to_sql(
                "daily_option_features",
                conn,
                if_exists="append",
                index=False,
                method="multi"
            )

        # d) Refresh the materialized view daily_features
        logger.info("Refreshing materialized view daily_features...")
        conn.execute(text("REFRESH MATERIALIZED VIEW daily_features;"))

    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()
