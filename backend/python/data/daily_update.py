#!/usr/bin/env python3
import os
import time
import datetime as dt
import logging
import yfinance as yf
import requests

import pandas as pd
from sqlalchemy import create_engine, text

# ─── Adjust these imports to point at your backfill module ──────────────────
from data.backfill_top100 import (
    fetch_stock_prices_yf,
    compute_price_features,
    fetch_vix_history,
    fetch_option_contracts_for_date,
    fetch_option_ohlcv,
    compute_option_greeks,
    build_option_features
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("Set POLYGON_API_KEY in your environment")

# DATABASE CONNECTION (adjust as needed)
DB_USER     = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "options_trading")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine      = create_engine(DB_URL, echo=False)

TWO_YEARS_AGO   = (dt.date.today() - pd.DateOffset(days=730)).date()
HIST_START_DATE = (dt.date.today() - pd.DateOffset(days=760)).date().isoformat()
START_DATE      = TWO_YEARS_AGO.isoformat()
END_DATE        = (dt.date.today())
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# ─────────────────────────────────────────────────────────────────────────────
# Ticker Utility
# ─────────────────────────────────────────────────────────────────────────────
def get_sp500_tickers() -> list:
    """
    Scrape the current S&P 500 tickers from Wikipedia.
    Returns a list of ticker strings.
    """
    tables = pd.read_html(SP500_WIKI_URL)
    df_sp500 = tables[0]
    tickers = df_sp500["Symbol"].tolist()
    # Replace dots (e.g. BRK.B) with dashes (BRK-B) for yfinance
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

def get_top_100_liquid_sp500() -> list:
    """
    Fetches the S&P 500 tickers, downloads their last 30 days of stock data via yfinance,
    sums the daily 'Volume' for each ticker, and returns the top 100 tickers by total volume.
    """
    # 1. Get all S&P 500 tickers
    sp500 = get_sp500_tickers()
    
    # 2. Download the last 30 calendar days of data for all tickers at once
    #    By default, yf.download returns a DataFrame with columns like:
    #      ('Open',   'AAPL'), ('High', 'AAPL'), …, ('Volume', 'AAPL'), etc.
    #    We only care about 'Volume'.
    data = yf.download(
        tickers=sp500,
        period="30d",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True
    )
    
    # 3. Extract the per-ticker 'Volume' DataFrame
    #    If only a single ticker is returned, yfinance returns a flat DataFrame.
    #    So handle both cases.
    try:
        vol_df = data.xs("Volume", axis=1, level=1)
    except KeyError:
        # In some edge cases (e.g. if yfinance returns a flat DataFrame),
        # handle the single-ticker or unexpected shape.
        # If it's a flat DataFrame with a "Volume" column:
        if "Volume" in data.columns:
            vol_df = data[["Volume"]].rename(columns={"Volume": sp500[0]})
        else:
            raise
    
    # 4. Sum the 30-day volume for each ticker
    total_vol = vol_df.sum(axis=0)
    
    # 5. Take the top 100 tickers by volume
    top100 = total_vol.sort_values(ascending=False).head(100).index.tolist()
    return top100

def last_trade_day(dt_end: dt.date) -> dt.date:
    # shift backwards until it's Mon–Fri
    while dt_end.weekday() >= 5:  # 5=Saturday,6=Sunday
        dt_end -= pd.Timedelta(days=1)
    return dt_end

def fetch_stock_for_options_after(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Grab all daily_stock_features rows for `ticker` on or after `start_date`.
    Returns columns: date, open, high, low, close, volume, plus whatever you need downstream.
    """
    sql = text("""
        SELECT
            trade_date     AS date,
            ticker         AS underlying_symbol,
            stk_open       AS open,
            stk_high       AS high,
            stk_low        AS low,
            stk_close      AS close,
            stk_volume     AS volume
          FROM daily_stock_features
         WHERE ticker      = :ticker
           AND trade_date >= :start_date
         ORDER BY trade_date
    """)
    df = pd.read_sql_query(
        sql,
        con=engine,
        params={"ticker": ticker, "start_date": start_date}
    )

    # cast to datetime and normalize to midnight
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df

def main():
    t0 = time.time()
    logging.info("===== STARTING BACKFILL PIPELINE =====")

    # first get today's date & a small history window for computing stock features
    last_trade_date = last_trade_day(END_DATE)
    last_trade_date_str  = last_trade_date.isoformat()

    # 1) grab the top‐100, stick any brand‐new ones into tickers…
    top100 = get_top_100_liquid_sp500()
    logging.info("Ensuring new top‐100 tickers are in 'tickers' table…")
    with engine.begin() as conn:
        for tk in top100:
            conn.execute(
                text("""
                  INSERT INTO tickers (ticker)
                  VALUES (:tk)
                  ON CONFLICT (ticker) DO NOTHING
                """),
                {"tk": tk}
            )
    logging.info("  ▶ New tickers added.")

    # 2) now build universe = _all_ tickers you've ever inserted (so we update both old & new)
    with engine.connect() as conn:
        universe = [row[0] for row in conn.execute(text("SELECT ticker FROM tickers")).all()]
    logging.info(f"Processing {len(universe)} total tickers (old & new)…")

    # ─── STEP A: DAILY VIX DATA ─────────────────────────────────────────────
    logging.info("Backfilling vix_data…")
    with engine.connect() as conn:
        last_vix_date = conn.execute(
            text("SELECT MAX(trade_date) FROM vix_data")
        ).scalar()
    if last_vix_date is None:
        vix_start = START_DATE
    else:
        # start the next calendar day
        vix_start = (last_vix_date + dt.timedelta(days=1)).isoformat()

    if last_vix_date is None or last_vix_date < last_trade_date:
        logging.info(f"Fetching VIX data from {vix_start} → {last_trade_date}")
        df_vix = fetch_vix_history(vix_start, last_trade_date_str)
        if not df_vix.empty:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        DELETE FROM vix_data
                        WHERE trade_date BETWEEN :start AND :end
                    """),
                    {"start": vix_start, "end": last_trade_date_str}
                )
                df_vix.to_sql(
                    "vix_data", conn,
                    if_exists="append", index=False, method="multi"
                )
            logging.info("  ▶ Inserted vix_data")

    # ─── STEP B: DAILY STOCK FEATURES ────────────────────────────────────────
    for sym in universe:
        with engine.connect() as conn:
            last_stock_date = conn.execute(
                text("SELECT MAX(trade_date) FROM daily_stock_features WHERE ticker = :tk"),
                {"tk": sym}
            ).scalar()
        if last_stock_date is None:
            stock_start = START_DATE
            hist_stock_start = HIST_START_DATE
        else:
            # start the next calendar day
            stock_start = (last_stock_date + dt.timedelta(days=1)).isoformat()
            hist_stock_start = (last_stock_date - dt.timedelta(days=30)).isoformat()

        if last_stock_date is None or last_stock_date < last_trade_date:
            df_stock_price_raw = fetch_stock_prices_yf(sym, hist_stock_start, last_trade_date_str)
            if df_stock_price_raw.empty:
                logging.warning(f"No raw stock data for {sym}. Skipping.")
                continue

            # Compute rolling stock features from raw OHLCV
            df_price_feats = compute_price_features(df_stock_price_raw)

            df_stock = df_stock_price_raw.merge(
                df_price_feats,
                how="left",
                on=["underlying_symbol", "date"]
            ).rename(columns={"underlying_symbol": "ticker", "date": "trade_date"})

            df_stock_final = df_stock[[
                "ticker", "trade_date",
                "open", "high", "low", "close", "volume",
                "realized_vol_10d", "price_change_3d",
                "sma_5", "sma_10", "ma5_ma10_diff", "atr_5",
                "rsi_14"
            ]].rename(columns={
                "open":   "stk_open",
                "high":   "stk_high",
                "low":    "stk_low",
                "close":  "stk_close",
                "volume": "stk_volume"
            })

            df_stock_final["stk_return_1d"]   = df_stock_final.groupby("ticker")["stk_close"].pct_change(1)
            df_stock_final["stk_range_pct"]   = (df_stock_final["stk_high"] - df_stock_final["stk_low"]) / (df_stock_final["stk_close"] + 1e-12)
            df_stock_final["stk_vol_chg_pct"] = df_stock_final.groupby("ticker")["stk_volume"].pct_change(1)

            df_stock_to_insert = df_stock_final[
                (df_stock_final["trade_date"] >= pd.to_datetime(stock_start)) &
                (df_stock_final["trade_date"] <= pd.to_datetime(last_trade_date_str))
            ].copy()

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        DELETE FROM daily_stock_features
                        WHERE trade_date BETWEEN :start AND :end
                          AND ticker = :tk
                    """),
                    {"start": stock_start, "end": last_trade_date_str, "tk": sym}
                )
                df_stock_to_insert.to_sql(
                    "daily_stock_features", conn,
                    if_exists="append", index=False, method="multi"
                )
            logging.info(f"  ▶ Inserted daily_stock_features for {sym}")

        # ─── STEP C: DAILY OPTION FEATURES ──────────────────────────────────────
    for sym in universe:
        with engine.connect() as conn:
            last_opt_date = conn.execute(
                text("SELECT MAX(trade_date) FROM daily_option_features WHERE ticker = :tk"),
                {"tk": sym}
            ).scalar()
        if last_opt_date is None:
            opt_start = START_DATE
        else:
            opt_start = (last_opt_date + dt.timedelta(days=1)).isoformat()

        if last_opt_date is None or last_opt_date < last_trade_date:
            # when you fetch contracts, only iterate dates from opt_start → yesterday
            option_trading_dates = pd.date_range(opt_start, last_trade_date_str).date
            all_contract_rows = []
            df_stock_for_options_raw = fetch_stock_for_options_after(sym, opt_start)
            for single_date in option_trading_dates:
                date_str = single_date.isoformat()
                underlying_close = df_stock_for_options_raw[
                    (df_stock_for_options_raw["underlying_symbol"] == sym) &
                    (df_stock_for_options_raw["date"].dt.date == single_date)
                ]["close"]
                if underlying_close.empty:
                    continue
                spot = float(underlying_close.iloc[0])
                min_strike = 0.8 * spot
                max_strike = 1.2 * spot

                rows = fetch_option_contracts_for_date(sym, date_str, min_strike, max_strike, POLYGON_API_KEY)
                all_contract_rows.extend(rows)
                logging.info(f"  ▶ Completed contract metadata for {date_str}")
            df_contracts = pd.DataFrame(all_contract_rows).drop_duplicates(subset=["option_symbol", "date"]).reset_index(drop=True)
            logging.info(f"  ▶ {len(df_contracts['option_symbol'].unique())} unique contracts for {sym}")

            # Fetch option OHLCV for filtered contracts
            logging.info("Fetching historical OHLCV for filtered option contracts…")
            option_syms = df_contracts['option_symbol'].unique().tolist()
            total       = len(option_syms)
            all_ohlcv_rows = []
            for idx, opt_sym in enumerate(option_syms, start=1):
                logging.info(f"  • Fetching option OHLCV ({idx}/{total}): {opt_sym}")
                df_ohlcv = fetch_option_ohlcv(opt_sym, opt_start, last_trade_date_str, POLYGON_API_KEY)
                if not df_ohlcv.empty:
                    all_ohlcv_rows.append(df_ohlcv)
            if all_ohlcv_rows:
                df_opt_ohlcv = pd.concat(all_ohlcv_rows, ignore_index=True)
            else:
                df_opt_ohlcv = pd.DataFrame(columns=[
                    "option_symbol", "date", "open", "high", "low", "close", "volume"
                ])
                logging.warning("  ⚠️  No option OHLCV data fetched; df_opt_ohlcv is empty.")
            logging.info(f"  ▶ Fetched OHLCV for {total} options of {sym}")

            logging.info("Computing IV + Greeks, merging stock OHLCV…")
            df_opt_greeks = compute_option_greeks(df_opt_ohlcv, df_stock_for_options_raw, df_contracts)
            logging.info(f"  ▶ Computed Greeks for options of {sym}")

            # Build option-level features (dropped low-importance columns, added nonlinear)
            df_ready = build_option_features(df_opt_greeks)

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        DELETE FROM daily_option_features
                        WHERE trade_date BETWEEN :start AND :end
                          AND option_symbol IN :opt_syms
                    """),
                    {
                        "start":    opt_start,
                        "end":      last_trade_date_str,
                        "opt_syms": tuple(df_ready["option_symbol"].unique())
                    }
                )
                try:
                  df_ready.to_sql(
                      "daily_option_features",
                      conn,
                      if_exists="append",
                      index=False,
                      method="multi"
                  )
                except Exception as e:
                    # Print only the exception text, not the full parameter list
                    print("Insert failed:", e)
                    # If you still want a minimal traceback, you can do:
                    # traceback.print_exception(type(e), e, e.__traceback__, limit=1)
            logging.info(f"  ▶ Inserted daily_option_features for {sym}")

    logging.info("✅ Daily injection complete.")
    logging.info("🎉 Backfill complete for all tickers, VIX, and options.")
    # ─────────────────────────────────────────────────────────────────────────
    t1 = time.time()
    logging.info(f"===== DONE! Total runtime: {round((t1 - t0)/60, 2)} minutes =====")

if __name__ == "__main__":
    main()
