from api.signal_generator.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY, TOP_TICKERS

import os
import logging
import requests
import datetime as dt
import pandas as pd

from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient

from concurrent.futures import ThreadPoolExecutor, as_completed

# Alpaca clients
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def load_broad_universe_from_csv():
    path = os.path.join(os.path.dirname(__file__), "../../resources/tickers.csv")
    df = pd.read_csv(path)
    return df["ticker"].dropna().unique().tolist()

def fetch_trending_tickers(top_n=5):
    tickers = load_broad_universe_from_csv()
    universe = [t for t in tickers if t not in TOP_TICKERS]
    logging.info(f"Loaded {len(universe)} tickers for trending scan.")

    def fetch_single(symbol):
        try:
            end = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(days=7)

            # Fetch daily bars
            bars = stock_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed='iex'
            )).df

            bars = bars.sort_index()
            if len(bars) < 2:
                return None

            prev_close = bars["close"].iloc[-2]
            latest = bars["close"].iloc[-1]
            pct_change = (latest - prev_close) / prev_close
            volume = bars["volume"].iloc[-1]

            return {
                "ticker": symbol,
                "pct_change": pct_change,
                "volume": volume
            }
        except Exception as e:
            logging.debug(f"Alpaca fetch failed for {symbol}: {e}")
            return None

    records = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single, t): t for t in universe}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    if not records:
        logging.warning("No trending data found from Alpaca.")
        return []

    df_out = pd.DataFrame(records)
    top_gainers = df_out.sort_values("pct_change", ascending=False).head(top_n)
    top_losers = df_out.sort_values("pct_change", ascending=True).head(top_n)
    top_volume = df_out.sort_values("volume", ascending=False).head(top_n)

    trending = pd.concat([top_gainers, top_losers, top_volume])["ticker"].dropna().unique().tolist()

    # Optional: save debug CSV
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame({"ticker": trending}).to_csv(f"trending_tickers_alpaca_{timestamp}.csv", index=False)
    logging.info(f"Wrote {len(trending)} trending tickers to trending_tickers_alpaca_{timestamp}.csv")

    return trending

def fetch_latest_data(ticker, start_date=None, end_date=None):
    try:
        if end_date is None:
            end = dt.datetime.now(dt.timezone.utc)
        else:
            end = dt.datetime.combine(end_date, dt.time(23, 59), tzinfo=dt.timezone.utc)
        if start_date is None:
            start = end - dt.timedelta(days=7)
        else:
            start = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc)

        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed='iex'
        )
        bars = stock_client.get_stock_bars(request_params).df
        logging.info(f"Fetched {len(bars)} bars for {ticker}")
        bars = bars.sort_index()

        quote_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = stock_client.get_stock_latest_quote(quote_req)[ticker]
        return bars, quote
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None, None
    
def fetch_tlt_return_5d():
    symbol = "TLT"
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=10)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed='iex'
    )
    bars = stock_client.get_stock_bars(request_params).df
    logging.info(f"Fetched {len(bars)} bars for {symbol}")
    bars = bars.sort_index()
    if len(bars) < 6:
        logging.warning("Not enough TLT data to compute 5-day return")
        return 0.0
    return bars['close'].iloc[-1] / bars['close'].iloc[-6] - 1

def fetch_option_chain(ticker, as_of_date):
    try:
        today = as_of_date
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=7)

        # Fetch recent close price from Alpaca
        bars = stock_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed='iex'
        )).df

        if bars.empty:
            logging.warning(f"No stock bars for {ticker}")
            return []

        spot = bars['close'].iloc[-1]
        min_strike = round(spot * 0.95, 2)
        max_strike = round(spot * 1.05, 2)

        base_url = "https://api.polygon.io/v3/reference/options/contracts"
        params   = {
            "underlying_ticker": ticker,
            "as_of":             today.isoformat(),
            "expiration_date.lte": (today + dt.timedelta(days=45)).isoformat(),
            "strike_price.gt":   min_strike,
            "strike_price.lt":   max_strike,
            "limit":             100,
            "apiKey":            POLYGON_API_KEY
        }

        contract_rows = []
        url = base_url

        while True:
            logging.info(f"    • Fetching contracts for {ticker} on {today} with strikes between {min_strike} and {max_strike}")
            resp = requests.get(url, params=params)
            if resp.status_code == 401:
                logging.error(f"      – Unauthorized (401) calling {url} params={params}")
                break
            if resp.status_code != 200:
                logging.warning(f"      – [Contracts] Failed {ticker} on {today}: HTTP {resp.status_code}")
                break

            data    = resp.json()
            results = data.get("results", [])
            contract_rows.extend(results)

            next_url = data.get("next_url")
            if not next_url:
                break
            if "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={POLYGON_API_KEY}"
            url    = next_url
            params = None

        logging.info(f"Polygon returned {len(contract_rows)} contracts for {ticker}")

        class PolygonContract:
            def __init__(self, raw):
                self.symbol = raw.get("ticker")
                self.strike_price = float(raw.get("strike_price", 0.0))
                self.option_type = raw.get("contract_type", "").lower()
                self.expiration_date = pd.to_datetime(raw.get("expiration_date")).date()

        return [PolygonContract(r) for r in contract_rows if pd.to_datetime(r.get("expiration_date")).date() >= today]

    except Exception as e:
        logging.error(f"Error fetching option contracts from Polygon for {ticker}: {e}")
        return []