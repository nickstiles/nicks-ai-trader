import os
import pandas as pd
import numpy as np
import joblib
import datetime as dt
import requests
import logging
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, OptionChainRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionIntent
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("signal_generator.log"),
        logging.StreamHandler()
    ]
)

DRY_RUN = True
CAPITAL_ALLOCATION_PCT = 0.05  # 5% per trade
MAX_OPEN_TRADES = 10
CONTRACT_MULTIPLIER = 100 
MIN_PRED_RETURN = 0.45
TRADE_LOG_FILE = "active_trades.csv"
STOP_LOSS_PCT = 0.30   # e.g., 30% loss triggers stop
TAKE_PROFIT_PCT = 0.50  # e.g., 50% gain triggers profit

# Load Alpaca credentials
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Alpaca clients
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
option_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Load models and beta vector
mean_model = joblib.load("mean_model.pkl")
quantile_models = joblib.load("quantile_models.pkl")
ranker_model = joblib.load("ranker_model.pkl")
meta_model = joblib.load("meta_model.pkl")
beta_vector = np.load("beta_vector.npy")  # shape: (2,)

# List of tickers (example: top 5)
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]

def load_active_trades():
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE, parse_dates=["entry_time"])
        return df[df["status"] == "open"].copy()
    return pd.DataFrame(columns=["option_symbol", "entry_time", "entry_price", "qty",
                                 "stop_price", "take_profit_price", "status"])

def append_trade_to_log(symbol, price, qty):
    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
    take_profit_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
    new_trade = pd.DataFrame([{
        "option_symbol": symbol,
        "entry_time": datetime.datetime.now(),
        "entry_price": price,
        "qty": qty,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "status": "open"
    }])
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        df = pd.concat([df, new_trade], ignore_index=True)
    else:
        df = new_trade
    df.to_csv(TRADE_LOG_FILE, index=False)

def manage_open_trades():
    active_trades = load_active_trades()
    if active_trades.empty:
        logging.info("No open trades to manage.")
        return

    for _, trade in active_trades.iterrows():
        symbol = trade["option_symbol"]
        qty = int(trade["qty"])
        stop = trade["stop_price"]
        target = trade["take_profit_price"]
        status = trade["status"]

        if status != "open":
            continue

        # Fetch current price
        try:
            url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
            resp = requests.get(url, timeout=3)
            data = resp.json().get("results", {})
            price = data.get("day", {}).get("close")
            if price is None:
                logging.warning(f"No price data for {symbol}")
                continue

            if price <= stop:
                logging.info(f"{symbol} hit stop-loss (${price:.2f} ≤ ${stop:.2f})")
                if not DRY_RUN:
                    trading_client.close_position(symbol=symbol)
                trade["status"] = "closed"

            elif price >= target:
                logging.info(f"{symbol} hit take-profit (${price:.2f} ≥ ${target:.2f})")
                if not DRY_RUN:
                    trading_client.close_position(symbol=symbol)
                trade["status"] = "closed"

        except Exception as e:
            logging.error(f"Failed to manage trade {symbol}: {e}")

    # Save updated trade log
    active_trades.to_csv(TRADE_LOG_FILE, index=False)

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

def fetch_latest_data(ticker):
    try:
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=7)
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
        

def compute_features(ticker):
    bars, quote = fetch_latest_data(ticker)
    if bars is None or len(bars) < 5:
        return None

    close_prices = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']

    latest_close = close_prices.iloc[-1]
    return {
        "tlt_return_5d": fetch_tlt_return_5d(),
        "stk_range_pct_zscore": ((high - low) / close_prices).rolling(5).mean().iloc[-1],
        "stoch_d": 100 * (latest_close - low.min()) / (high.max() - low.min() + 1e-6),
        "mid_price": latest_close,
        "stock_price": latest_close,
    }

def fetch_option_chain(ticker):
    try:
        today = dt.datetime.now().date()
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
        min_strike = round(spot * 0.8, 2)
        max_strike = round(spot * 1.2, 2)

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

def build_feature_rows(ticker):
    feats = compute_features(ticker)
    if feats is None:
        logging.warning(f"Insufficient feature data for {ticker}, skipping.")
        return []

    contracts = fetch_option_chain(ticker)
    if not contracts:
        return []

    polygon_symbols = []
    contract_map = {}
    for c in contracts:
        try:
            expiry = dt.datetime.strptime(str(c.expiration_date), "%Y-%m-%d").strftime("%y%m%d")
            cp = "C" if c.option_type == "call" else "P"
            strike = int(float(c.strike_price) * 1000)
            poly_symbol = f"O:{ticker}{expiry}{cp}{strike:08d}"
            polygon_symbols.append(poly_symbol)
            contract_map[poly_symbol] = c
        except Exception as e:
            logging.warning(f"Failed to parse contract for {ticker}: {e}")
            continue

    logging.info(f"Built {len(polygon_symbols)} Polygon symbols for {ticker}")

    def fetch_snapshot(symbol):
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}/{symbol}?apiKey={POLYGON_API_KEY}"
        try:
            resp = requests.get(url, timeout=3)
            return symbol, resp.json().get("results", {})
        except Exception as e:
            logging.debug(f"Error fetching {symbol}: {e}")
            return symbol, None

    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch_snapshot, s) for s in polygon_symbols]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                logging.debug(f"No snapshot data for {symbol}")
                continue

            delta = data.get("greeks", {}).get("delta")
            iv = data.get("implied_volatility")
            mid_price = data.get("day", {}).get("close")

            if delta is None or iv is None or mid_price is None:
                logging.debug(f"Missing greeks or mid-price for {symbol}")
                continue

            contract = contract_map[symbol]
            dte = (contract.expiration_date - dt.datetime.now().date()).days
            if dte < 3 or dte > 45:
                logging.debug(f"DTE {dte} out of range for {symbol}")
                continue

            moneyness = feats['stock_price'] / contract.strike_price
            if moneyness < 0.8 or moneyness > 1.2:
                logging.debug(f"Moneyness {moneyness:.2f} out of range for {symbol}")
                continue

            delta_iv = delta * iv
            parent_vec = np.array([delta, moneyness])
            y_pred = parent_vec.dot(beta_vector)
            y_actual = delta * moneyness
            resid = y_actual - y_pred

            row = {
                "ticker": ticker,
                "option_symbol": contract.symbol,
                "option_price": mid_price,
                "strike": contract.strike_price,
                "expiry": contract.expiration_date,
                "delta": delta,
                "moneyness": moneyness,
                "tlt_return_5d": feats['tlt_return_5d'],
                "delta_iv": delta_iv,
                "stoch_d": feats['stoch_d'],
                "stk_range_pct_zscore": feats['stk_range_pct_zscore'],
                "implied_volatility": np.log1p(iv),
                "delta_x_moneyness_resid": resid,
            }
            rows.append(row)

    logging.info(f"{ticker} \\u2192 {len(rows)} valid options after filtering")
    return rows

def get_account_equity():
    try:
        acct = trading_client.get_account()
        equity = float(acct.equity)
        logging.info(f"Current account equity: ${equity:,.2f}")
        return equity
    except Exception as e:
        logging.error(f"Failed to fetch account equity: {e}")
        return 0.0

def calculate_contract_qty(option_price, equity):
    capital_per_trade = equity * CAPITAL_ALLOCATION_PCT
    if option_price <= 0:
        return 0
    total_cost_1_contract = option_price * CONTRACT_MULTIPLIER
    if total_cost_1_contract > capital_per_trade:
        return 0
    return int(capital_per_trade / total_cost_1_contract)

def submit_option_order(option_symbol, option_price, qty):
    if DRY_RUN:
        logging.info(f"[DRY RUN] Would submit option order: BUY_TO_OPEN {qty}x {option_symbol} @ ${option_price:.2f}")
        append_trade_to_log(option_symbol, option_price, qty)
        return None

    legs = [
        OptionLegRequest(
            symbol=option_symbol,
            ratio_qty=qty,
            side=OrderSide.BUY,
            position_intent=PositionIntent.OPEN
        )
    ]

    order_data = MarketOrderRequest(
        symbol=option_symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        legs=legs
    )

    try:
        order = trading_client.submit_order(order_data=order_data)
        logging.info(f"Submitted option order: ID {order.id}")
        append_trade_to_log(option_symbol, option_price, qty)
        return order
    except Exception as e:
        logging.error(f"Failed to submit option order for {option_symbol}: {e}")
        return None


def run_signal_generation():
    logging.info("Starting signal generation")
    results = []
    for ticker in tickers:
        logging.info(f"Processing {ticker}...")
        rows = build_feature_rows(ticker)
        for row in rows:
            df = pd.DataFrame([{
                k: row[k] for k in [
                    "delta", "moneyness", "tlt_return_5d", "delta_iv",
                    "stoch_d", "stk_range_pct_zscore", "implied_volatility",
                    "delta_x_moneyness_resid"
                ]
            }])
            try:
                meta_input = np.vstack([
                    mean_model.predict(df),
                    quantile_models[0.1].predict(df),
                    quantile_models[0.5].predict(df),
                    quantile_models[0.9].predict(df),
                    ranker_model.predict(df)
                ]).T
                pred = meta_model.predict(meta_input)[0]
                row["pred_return"] = pred
                results.append(row)
            except Exception as e:
                logging.error(f"Model prediction failed for {row['option_symbol']}: {e}")

    # Filter overlapping signals: keep only top per underlying per expiry
    filtered = {}
    for row in sorted(results, key=lambda x: -x["pred_return"]):
        key = (row["ticker"], row["expiry"])
        if key not in filtered:
            filtered[key] = row
    filtered_signals = list(filtered.values())

    # Filter to only those predicted to be in the top 10% using q10 threshold
    for row in filtered_signals:
        df = pd.DataFrame([{k: row[k] for k in [
            "delta", "moneyness", "tlt_return_5d", "delta_iv",
            "stoch_d", "stk_range_pct_zscore", "implied_volatility",
            "delta_x_moneyness_resid"
        ]}])
        q10_pred = quantile_models[0.1].predict(df)[0]
        row["q10_pred"] = q10_pred

    final_signals = [
        r for r in filtered_signals
        if r["pred_return"] > r["q10_pred"] and r["pred_return"] >= MIN_PRED_RETURN
    ]

    # Show top signals
    top = sorted(final_signals, key=lambda x: -x["pred_return"])[:10]
    print("Top Signals:")
    for row in top:
        print(row)

    # Save only sorted final signals to CSV
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"signals_{ts}.csv"
    pd.DataFrame(sorted(final_signals, key=lambda x: -x["pred_return"])).to_csv(out_path, index=False)
    logging.info(f"Saved {len(final_signals)} filtered signals to {out_path}")

    final_signals_df = pd.DataFrame(final_signals)
    final_signals_df = (
        final_signals_df
        .sort_values(by='pred_return', ascending=False)
        .groupby('ticker')
        .head(2)
    )

    equity = get_account_equity()

    for signal in final_signals_df.to_dict("records"):
        max_alloc = equity * CAPITAL_ALLOCATION_PCT
        price = signal.get("option_price")
        if price is None or price <= 0:
            logging.warning(f"Skipping {signal['option_symbol']} due to missing or invalid price")
            continue
        
        qty = calculate_contract_qty(price, equity)
        if qty == 0:
            logging.warning(f"Skipping {signal['option_symbol']}: option too expensive for allocation")
            continue

        total_cost = price * qty * CONTRACT_MULTIPLIER
        logging.debug(f"{signal['option_symbol']} total cost = ${total_cost:.2f}, allowed = ${max_alloc:.2f}")
        if total_cost > max_alloc:
            logging.warning(
                f"Skipping {signal['option_symbol']}: cost ${total_cost:.2f} exceeds allocation (${max_alloc:.2f})"
            )
            continue

        submit_option_order(signal["option_symbol"], price, qty)
        equity -= price * qty * CONTRACT_MULTIPLIER

if __name__ == "__main__":
    run_signal_generation()