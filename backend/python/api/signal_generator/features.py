from api.signal_generator.config import POLYGON_API_KEY
from api.signal_generator.models import load_beta_vector
from api.signal_generator.fetchers import fetch_latest_data, fetch_tlt_return_5d, fetch_option_chain

import logging
import numpy as np
import datetime as dt
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

# Load beta vector
beta_vector = load_beta_vector()

def compute_features(ticker, tlt_return_5d, as_of_date, lookback_start):
    bars, quote = fetch_latest_data(ticker, start_date=lookback_start, end_date=as_of_date)
    if bars is None or len(bars) < 5:
        return None

    close_prices = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']

    latest_close = close_prices.iloc[-1]
    return {
        "tlt_return_5d": tlt_return_5d,
        "stk_range_pct_zscore": ((high - low) / close_prices).rolling(5).mean().iloc[-1],
        "stoch_d": 100 * (latest_close - low.min()) / (high.max() - low.min() + 1e-6),
        "stock_price": latest_close,
    }

def build_feature_rows(ticker, tlt_return_5d, as_of_date, lookback_start):
    feats = compute_features(ticker, tlt_return_5d, as_of_date, lookback_start)
    if feats is None:
        logging.warning(f"Insufficient feature data for {ticker}, skipping.")
        return []

    contracts = fetch_option_chain(ticker, as_of_date)
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

            volume = data.get("day", {}).get("volume", 0)
            if volume < 5:
                logging.debug(f"Volume {volume} too low for {symbol}")
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
            if moneyness < 0.95 or moneyness > 1.05:
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

    logging.info(f"{ticker} has {len(rows)} valid options after filtering")
    return rows