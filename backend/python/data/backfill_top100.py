import os
import time
import datetime as dt
import logging

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from scipy.stats import norm
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("Set POLYGON_API_KEY in your environment")

# 3) EXPANDED TRAINING WINDOW → 12 months
TWO_YEARS_AGO = (dt.date.today() - pd.DateOffset(days=730)).date()
START_DATE        = TWO_YEARS_AGO.isoformat()
END_DATE          = (dt.date.today() - pd.DateOffset(days=1)).date().isoformat()

# DATABASE CONNECTION (adjust as needed)
DB_USER     = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "options_trading")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine      = create_engine(DB_URL, echo=False)

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

# ─────────────────────────────────────────────────────────────────────────────
# BLACK‐SCHOLES + GREEKS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-12)

def bs_price(S, K, T, r, sigma, option_type):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_implied_vol(option_mid, S, K, T, r, option_type, tol=1e-6, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-8:
            break
        diff = price - option_mid
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return np.nan

def compute_bs_greeks(S, K, T, r, sigma, option_type):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    if option_type.lower() == "call":
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2)
    else:
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    gamma = pdf_d1 / (S * sigma * np.sqrt(T) + 1e-12)
    vega = S * pdf_d1 * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FETCH RAW STOCK OHLCV via yfinance
# ─────────────────────────────────────────────────────────────────────────────
def fetch_stock_prices_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    end_inc = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end_inc,
        interval="1d",
        progress=False
    ).reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw.columns = ["date", "open", "high", "low", "close", "volume"]
    raw["underlying_symbol"] = ticker.upper()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["date"] = raw["date"].dt.normalize()
    return raw[["underlying_symbol", "date", "open", "high", "low", "close", "volume"]]

def fetch_vix_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily VIX (^VIX) history between start_date and end_date,
    returning ['date','vix_open','vix_high','vix_low','vix_close','vix_volume'].
    """
    vix_ticker = "^VIX"
    raw = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)

    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df_v = raw.reset_index().rename(columns={
        "Date":   "date",
        "Open":   "vix_open",
        "High":   "vix_high",
        "Low":    "vix_low",
        "Close":  "vix_close",
        "Volume": "vix_volume"
    })[["date", "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume"]]

    df_v["date"]       = pd.to_datetime(df_v["date"]).dt.normalize()
    df_v["trade_date"] = df_v["date"].dt.date

    start_dt = pd.to_datetime(start_date).date()
    end_dt   = pd.to_datetime(end_date).date()
    df_v = df_v[
        (df_v["trade_date"] >= start_dt) &
        (df_v["trade_date"] <= end_dt)
    ].copy()

    return df_v[["trade_date", "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume"]]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE ROLLING STOCK FEATURES (added sma_5, sma_10, ma5_ma10_diff, atr_5)
# ─────────────────────────────────────────────────────────────────────────────
def compute_price_features(df_price: pd.DataFrame) -> pd.DataFrame:
    df_price = df_price.sort_values(["underlying_symbol", "date"]).copy()
    out = []
    for sym, grp in df_price.groupby("underlying_symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["log_ret"]          = np.log(grp["close"] / grp["close"].shift(1))
        grp["realized_vol_10d"] = grp["log_ret"].rolling(10).std() * np.sqrt(252)
        grp["price_change_3d"]  = grp["close"] / grp["close"].shift(3) - 1

        # New: SMA 5, SMA 10, their diff, ATR 5
        grp["sma_5"]         = grp["close"].rolling(5).mean()
        grp["sma_10"]        = grp["close"].rolling(10).mean()
        grp["ma5_ma10_diff"] = grp["sma_5"] - grp["sma_10"]
        grp["atr_5"]         = (grp["high"].rolling(5).max() - grp["low"].rolling(5).min()) / (grp["close"] + 1e-12)

        # RSI 14
        delta_   = grp["close"].diff()
        gain     = delta_.clip(lower=0)
        loss     = -delta_.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs       = avg_gain / (avg_loss + 1e-8)
        grp["rsi_14"] = 100 - (100 / (1 + rs))

        out.append(
            grp[[
                "underlying_symbol",
                "date",
                "realized_vol_10d",
                "price_change_3d",
                "sma_5", "sma_10", "ma5_ma10_diff", "atr_5",
                "rsi_14"
            ]]
        )
    df_feats = pd.concat(out, ignore_index=True)
    logging.info(f"  ▶ Stock‐feature rows: {len(df_feats)}")
    return df_feats

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FETCH OPTION CONTRACTS METADATA WITH STRIKE FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_option_contracts_for_date(underlying_symbol: str, date: str,
                                    min_strike: float, max_strike: float, api_key: str) -> list:
    base_url = "https://api.polygon.io/v3/reference/options/contracts"
    params   = {
        "underlying_ticker": underlying_symbol,
        "as_of":             date,
        "strike_price.gt":   min_strike,
        "strike_price.lt":   max_strike,
        "limit":             100,
        "apiKey":            api_key
    }

    contract_rows = []
    url = base_url

    while True:
        logging.info(
            f"    • Fetching contracts for {underlying_symbol} on {date} "
            f"with strikes between {min_strike:.2f} and {max_strike:.2f}"
        )
        resp = requests.get(url, params=params)
        if resp.status_code == 401:
            logging.error(f"      – Unauthorized (401) calling {url} params={params}")
            break
        if resp.status_code != 200:
            logging.warning(f"      – [Contracts] Failed {underlying_symbol} on {date}: HTTP {resp.status_code}")
            break

        data    = resp.json()
        results = data.get("results", [])
        for r in results:
            contract_rows.append({
                "underlying_symbol":  underlying_symbol,
                "option_symbol":      r.get("ticker"),
                "strike":             float(r.get("strike_price", np.nan)),
                "expiration_date":    pd.to_datetime(r.get("expiration_date")),
                "option_type":        r.get("contract_type"),
                "date":               pd.to_datetime(date)  # keep as datetime64
            })

        next_url = data.get("next_url")
        if not next_url:
            break

        if "apiKey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}apiKey={api_key}"
        url    = next_url
        params = None

    return contract_rows

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FETCH HISTORICAL OPTION OHLCV FOR EACH CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
def fetch_option_ohlcv(option_symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{option_symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "false", "sort": "asc", "apiKey": api_key}

    max_retries = 3
    backoff = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                rows = []
                for bar in data:
                    date = pd.to_datetime(bar["t"], unit="ms")
                    rows.append({
                        "option_symbol": option_symbol,
                        "date":          date,
                        "open":          bar["o"],
                        "high":          bar["h"],
                        "low":           bar["l"],
                        "close":         bar["c"],
                        "volume":        bar["v"]
                    })
                return pd.DataFrame(rows)

            # Non-200 but not a connection error
            logging.warning(f"[OHLCV] {option_symbol}: HTTP {resp.status_code}")
            return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            # ConnectionError, Timeout, RemoteDisconnected, etc.
            logging.warning(f"[OHLCV] {option_symbol}: attempt {attempt} failed → {e!r}")
            if attempt == max_retries:
                logging.error(f"[OHLCV] {option_symbol}: giving up after {max_retries} attempts")
                return pd.DataFrame()
            time.sleep(backoff)
            backoff *= 2

    # Should never reach here
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: COMPUTE GREEKS + ATM SKEW + ΔSKEW, MERGE RAW STOCK OHLCV
# ─────────────────────────────────────────────────────────────────────────────
def compute_option_greeks(
    df_opt_ohlcv: pd.DataFrame,
    df_stock_price_raw: pd.DataFrame,
    df_contracts: pd.DataFrame
) -> pd.DataFrame:
    """
    1) Align and merge raw option OHLCV, contract metadata, and stock OHLCV.
    2) Drop rows with missing data, zero/negative time to expiration.
    3) Drop rows with mid_price ≤ $0.05 or stk_close ≤ 0.
    4) Compute IV + Greeks row-by-row.
    5) Drop any rows where IV inversion failed (NaN).
    6) Compute 1-day IV change, ATM skew, delta_skew, and skew_vol.
    """
    risk_free_rate = 0.01

    # 1) Normalize dates on option OHLCV
    df_opt = df_opt_ohlcv.copy()
    df_opt["date"] = (
        pd.to_datetime(df_opt["date"], errors="coerce")
          .dt.tz_localize(None)
          .dt.normalize()
    )

    # 2) Rename option OHLCV columns
    if "volume" in df_opt.columns:
        df_opt = df_opt.rename(columns={
            "open":   "opt_open",
            "high":   "opt_high",
            "low":    "opt_low",
            "close":  "opt_close",
            "volume": "opt_volume"
        })
    else:
        df_opt["opt_volume"] = 0

    # 3) Merge contract metadata
    df_contracts_clean = df_contracts.drop(columns=["date"], errors="ignore")
    df = df_opt.merge(
        df_contracts_clean.drop_duplicates(subset=["option_symbol"]),
        on="option_symbol",
        how="left"
    )
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")

    # 4) Merge stock OHLCV for greeks inputs
    df = df.merge(
        df_stock_price_raw[["underlying_symbol", "date", "close", "volume"]]
            .rename(columns={
                "close":  "stk_close",
                "volume": "stk_volume"
            }),
        on=["underlying_symbol", "date"],
        how="left"
    )

    # 5) Drop rows missing essential data
    before = len(df)
    df = df.dropna(subset=["stk_close", "strike", "option_type"]).reset_index(drop=True)
    logging.info(f"  ▶ Dropped {before - len(df)} rows missing stock/options metadata")

    # 6) Compute time to expiry and drop non-positive T
    df["T"] = ((df["expiration_date"] - df["date"]).dt.days.clip(lower=0) / 365.0)
    before = len(df)
    df = df[df["T"] > 0].reset_index(drop=True)
    logging.info(f"  ▶ Dropped {before - len(df)} rows with T ≤ 0")

    # 7) Set mid_price
    df["mid_price"] = df["opt_close"]

    # ── DROP rows with too-low price or non-positive stock ───────────────
    before = len(df)
    df = df[
        (df["mid_price"] > 0.05) &
        (df["stk_close"] > 0)
    ].reset_index(drop=True)
    logging.info(f"  ▶ Dropped {before - len(df)} rows with mid_price ≤ $0.05 or stk_close ≤ 0")

    # 8) Compute IV + Greeks
    iv_list, delta_list, gamma_list, theta_list, vega_list = [], [], [], [], []
    for _, row in df.iterrows():
        S, K, T, r = row["stk_close"], row["strike"], row["T"], risk_free_rate
        price, opt_type = row["mid_price"], row["option_type"]

        if price <= 0.05 or S <= 0:
            iv = 2.0
            greeks = compute_bs_greeks(S, K, T, r, iv, opt_type)
        else:
            iv = bs_implied_vol(price, S, K, T, r, opt_type)
            if not np.isfinite(iv) or iv <= 0:
                iv = 2.0
                greeks = compute_bs_greeks(S, K, T, r, iv, opt_type)
            else:
                greeks = compute_bs_greeks(S, K, T, r, iv, opt_type)

        iv_list.append(iv)
        delta_list.append(greeks["delta"])
        gamma_list.append(greeks["gamma"])
        theta_list.append(greeks["theta"])
        vega_list.append(greeks["vega"])

    df["implied_volatility"] = iv_list
    df["delta"]              = delta_list
    df["gamma"]              = gamma_list
    df["theta"]              = theta_list
    df["vega"]               = vega_list

    # ── DROP rows where IV inversion failed ──────────────────────────────
    before = len(df)
    df = df[df["implied_volatility"].notna()].reset_index(drop=True)
    logging.info(f"  ▶ Dropped {before - len(df)} rows with missing IV")

    # 9) IV 1-day change
    df = df.sort_values(["option_symbol", "date"]).reset_index(drop=True)
    df["iv_change_1d"] = df.groupby("option_symbol")["implied_volatility"].pct_change(1)

    # 10) ATM skew and delta_skew
    def _atm_skew(sub: pd.DataFrame) -> float:
        spot = sub["stk_close"].iloc[0]
        plus = (sub["strike"] - spot * 1.05).abs().idxmin()
        minus= (sub["strike"] - spot * 0.95).abs().idxmin()
        return sub.loc[plus, "implied_volatility"] - sub.loc[minus, "implied_volatility"]

    atm_skew = (
        df[["underlying_symbol","date","strike","implied_volatility","stk_close"]]
          .groupby(["underlying_symbol","date"])
          .apply(_atm_skew)
          .reset_index(name="iv_skew_atm")
    )
    atm_skew["delta_skew"] = atm_skew.groupby("underlying_symbol")["iv_skew_atm"].diff().fillna(0)
    df = df.merge(atm_skew, on=["underlying_symbol","date"], how="left")
    df["skew_vol"] = df["iv_skew_atm"] * df["stk_volume"]

    # 11) Select final columns
    return df[[
        "option_symbol","underlying_symbol","date","expiration_date",
        "strike","option_type","mid_price","implied_volatility",
        "iv_change_1d","delta","gamma","theta","vega","opt_volume",
        "opt_open","opt_high","opt_low","opt_close",
        "stk_close","stk_volume",
        "iv_skew_atm","delta_skew","skew_vol"
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: BUILD OPTION FEATURES (drop low‐importance columns, add ATM skew + Δskew + interactions)
# ─────────────────────────────────────────────────────────────────────────────
def build_option_features(df_sel: pd.DataFrame) -> pd.DataFrame:
    df = df_sel.copy()

    # Ensure essential columns exist (including the new skew columns)
    needed = [
        "implied_volatility", "delta", "gamma", "theta", "vega",
        "iv_skew_atm", "skew_vol"  # ATM skew + its one-day diff
    ]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    logging.info(f"  ▶ Dropped {before - len(df)} rows due to missing essential data")

    # 1) Nonlinear time feature
    df["dte_days"]  = (df["expiration_date"] - df["date"]).dt.days.clip(lower=0)
    df["moneyness"] = df["stk_close"] / df["strike"]
    df["delta_sqrt_t"] = df["delta"] * np.sqrt(df["dte_days"] + 1e-12)

    # 2) Interaction features
    df["delta_gamma"] = df["delta"] * df["gamma"]
    df["delta_iv"]    = df["delta"] * df["implied_volatility"]

    # 3) Select final features (dropping low‐importance columns)
    feature_cols = [
        # GREKS & IV & ATM SKEW
        "delta", "gamma", "theta", "vega", 
        "implied_volatility", "iv_skew_atm", "skew_vol",
        # Interactions
        "delta_gamma", "delta_iv",
        # Time & moneyness
        "dte_days", "moneyness",
    ]

    # Include stk_close for baseline
    df_model = df[[
        "option_symbol", "underlying_symbol", "date", "opt_volume",
        "opt_open", "opt_high", "opt_low", "opt_close",
        "expiration_date", "strike", "option_type"
    ] + feature_cols]

    df_model = df_model.rename(columns={"underlying_symbol": "ticker", "date": "trade_date"})
    required_cols = [
        "ticker",
        "trade_date",
        "implied_volatility",
        "delta",
        "dte_days",
        "moneyness",
        "option_symbol",
        "option_type",
        "strike",
        "expiration_date"
    ]
    df_model = df_model.dropna(subset=required_cols).reset_index(drop=True)

    logging.info(f"  ▶ Final Feature DataFrame shape (with ATM skew): {df_model.shape}")
    return df_model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    logging.info("===== STARTING BACKFILL PIPELINE =====")
    universe = get_top_100_liquid_sp500()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT ticker
              FROM daily_option_features
        """))
        done = {row[0] for row in result}
    to_do = [tk for tk in universe if tk not in done]
    logging.info(f"Tickers already completed in daily_option_features: {sorted(done)}")
    logging.info(f"Will backfill options for: {to_do}")
    logging.info("Ensuring all underlyings are in 'tickers' table…")
    with engine.begin() as conn:
        for sym in to_do:
            conn.execute(
                text("""
                    INSERT INTO tickers (ticker)
                    VALUES (:tk)
                    ON CONFLICT (ticker) DO NOTHING
                """),
                {"tk": sym}
            )
    logging.info("  ▶ Tickers ensured.")

    # STEP 1: BACKFILL VIX_DATA (once)
    logging.info("Backfilling vix_data…")
    df_vix = fetch_vix_history(START_DATE, END_DATE)
    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM vix_data
                 WHERE trade_date BETWEEN :start AND :end
            """),
            {"start": START_DATE, "end": END_DATE}
        )
        df_vix.to_sql(
            "vix_data", conn,
            if_exists="append", index=False, method="multi"
        )
    logging.info("  ▶ Inserted vix_data")

    # Now loop per‐ticker
    for sym in to_do:
        logging.info(f"===== PROCESSING {sym} =====")
        # ─── STEP A: FETCH + INSERT DAILY_STOCK_FEATURES for sym ───────────────
        df_stock_price_raw = fetch_stock_prices_yf(sym, START_DATE, END_DATE)
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

        with engine.begin() as conn:
            conn.execute(
                text("""
                    DELETE FROM daily_stock_features
                     WHERE trade_date BETWEEN :start AND :end
                       AND ticker = :tk
                """),
                {"start": START_DATE, "end": END_DATE, "tk": sym}
            )
            df_stock_final.to_sql(
                "daily_stock_features", conn,
                if_exists="append", index=False, method="multi"
            )
        logging.info(f"  ▶ Inserted daily_stock_features for {sym}")

        # ─── STEP B: FETCH OPTION METADATA + OHLCV + GREEKS + FEATURES for sym ──
        logging.info(f"Fetching option contracts metadata for {sym} with strike filters…")
        trading_dates = sorted(df_stock_price_raw["date"].dt.date.unique())
        all_contract_rows = []
        for single_date in trading_dates:
            date_str = single_date.isoformat()
            underlying_close = df_stock_price_raw[
                (df_stock_price_raw["underlying_symbol"] == sym) &
                (df_stock_price_raw["date"].dt.date == single_date)
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

        # 4) Fetch option OHLCV for filtered contracts
        logging.info("Step 4: Fetching historical OHLCV for filtered option contracts…")
        option_syms = df_contracts['option_symbol'].unique().tolist()
        total       = len(option_syms)
        all_ohlcv_rows = []
        for idx, opt_sym in enumerate(option_syms, start=1):
            logging.info(f"  • Fetching option OHLCV ({idx}/{total}): {opt_sym}")
            df_ohlcv = fetch_option_ohlcv(opt_sym, START_DATE, END_DATE, POLYGON_API_KEY)
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

        logging.info("Step 5: Computing IV + Greeks, merging stock OHLCV…")
        df_opt_greeks = compute_option_greeks(df_opt_ohlcv, df_stock_price_raw, df_contracts)
        logging.info(f"  ▶ Computed Greeks for options of {sym}")

        # 7) Build option-level features (dropped low-importance columns, added nonlinear)
        df_ready = build_option_features(df_opt_greeks)

        with engine.begin() as conn:
            conn.execute(
                text("""
                    DELETE FROM daily_option_features
                     WHERE trade_date BETWEEN :start AND :end
                       AND option_symbol IN :opt_syms
                """),
                {
                    "start":    START_DATE,
                    "end":      END_DATE,
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

    logging.info("🎉 Backfill complete for all tickers, VIX, and options.")
    # ─────────────────────────────────────────────────────────────────────────
    t1 = time.time()
    logging.info(f"===== DONE! Total runtime: {round((t1 - t0)/60, 2)} minutes =====")

if __name__ == "__main__":
    main()
