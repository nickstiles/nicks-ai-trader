from api.signal_generator.config import POLYGON_API_KEY

import requests
import logging

def get_option_snapshot(ticker: str, option_symbol: str):
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}/{option_symbol}?apiKey={POLYGON_API_KEY}"
    try:
        resp = requests.get(url, timeout=3)
        data = resp.json().get("results", {})
        return data.get("day", {}).get("close", None)
    except Exception as e:
        logging.warning(f"Failed to fetch snapshot for {option_symbol}: {e}")
        return None