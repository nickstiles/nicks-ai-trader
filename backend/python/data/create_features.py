import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import ta

from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# DATABASE CONNECTION (adjust as needed)
DB_USER     = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise RuntimeError("DB_PASSWORD must be set in your environment")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "options_trading")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine      = create_engine(DB_URL, echo=False)

# ─────────────────────────────────────────────────────────────────────────────
# MACRO FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_vix_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily VIX (^VIX) history between start_date and end_date,
    returning ['date','vix_open','vix_high','vix_low','vix_close','vix_volume'].
    """
    vix_ticker = "^VIX"
    raw = yf.download(vix_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df_v = raw.reset_index().rename(columns={
        "Date":   "trade_date",
        "Open":   "vix_open",
        "High":   "vix_high",
        "Low":    "vix_low",
        "Close":  "vix_close",
        "Volume": "vix_volume"
    })[["trade_date", "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume"]]

    df_v["trade_date"] = pd.to_datetime(df_v["trade_date"]).dt.normalize()

    return df_v[["trade_date", "vix_open", "vix_high", "vix_low", "vix_close", "vix_volume"]]

def fetch_macro_df(start_date: str, end_date: str) -> pd.DataFrame:

    def fetch_price(ticker: str, col_prefix: str) -> pd.DataFrame:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)  # flatten columns

        if raw.empty or "Close" not in raw.columns:
            raise RuntimeError(f"Failed to fetch or parse {ticker}")

        df = raw.reset_index().rename(columns={
            "Date": "trade_date",
            "Close": f"{col_prefix}_close"
        })[["trade_date", f"{col_prefix}_close"]]

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()

        return df[["trade_date", f"{col_prefix}_close"]]

    # Fetch SPY and TLT
    df_spy = fetch_price("SPY", "spy")
    df_tlt = fetch_price("TLT", "tlt")

    # Compute 5-day returns
    df_spy["spy_return_5d"] = df_spy["spy_close"].pct_change(5)
    df_tlt["tlt_return_5d"] = df_tlt["tlt_close"].pct_change(5)

    # Fetch VIX using your function
    df_vix = fetch_vix_history(start_date, end_date)[["trade_date", "vix_close"]]

    # Merge step-by-step
    macro_base = df_spy[["trade_date", "spy_return_5d"]].merge(
        df_tlt[["trade_date", "tlt_return_5d"]], on="trade_date", how="inner"
    )

    macro_df = (
        macro_base
        .merge(df_vix[["trade_date", "vix_close"]], on="trade_date", how="inner")
        .dropna()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    return macro_df

def add_macro_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro features (spy_return_5d, tlt_return_5d) and attach vix_percentile_rank
    calculated from macro_df. Assumes df does NOT already have vix_close.
    """
    df = df.copy()

    # Merge SPY and TLT returns only
    df = df.merge(
        macro_df[["trade_date", "spy_return_5d", "tlt_return_5d"]],
        on="trade_date",
        how="left"
    )

    # Calculate VIX percentile rank from macro_df's vix_close
    vix_ref = macro_df[["trade_date", "vix_close"]].dropna().sort_values("trade_date").copy()
    vix_ref["vix_percentile_rank"] = (
        vix_ref["vix_close"]
        .rolling(window=60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    df = df.merge(vix_ref[["trade_date", "vix_percentile_rank"]], on="trade_date", how="left")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def pick_nearest_iv(df: pd.DataFrame, target_dte: int, name: str) -> pd.DataFrame:
    """
    For each date and underlying_symbol, pick the implied_volatility
    from the option whose dte_days is closest to target_dte.
    """
    nearest = (
        df
        .assign(diff=lambda d: (d['dte_days'] - target_dte).abs())
        .sort_values(['trade_date', 'underlying_symbol', 'diff'])
        .groupby(['trade_date', 'underlying_symbol'], as_index=False)
        .first()[['trade_date', 'underlying_symbol', 'implied_volatility']]
        .rename(columns={'implied_volatility': name})
    )
    return nearest

def add_vol_term_structure_and_calendar_skew(
    df: pd.DataFrame,
    dte_short: int = 7,
    dte_long: int  = 30,
    atm_tol: float = 0.05
) -> pd.DataFrame:
    """
    Enhance df with live-safe term-structure & calendar skew features using nearest-expiry lookup.
    - iv_{dte_short}d, iv_{dte_long}d: implied vol at the tenor closest to target.
    - vol_term_structure = iv_long - iv_short
    - atm_iv_{dte_short}d, atm_iv_{dte_long}d: as above but only ATM strikes.
    - calendar_skew = atm_iv_long - atm_iv_short
    """
    # 1) Nearest-IV for short & long tenors
    short_df = pick_nearest_iv(df, dte_short, f'iv_{dte_short}d')
    long_df  = pick_nearest_iv(df, dte_long,  f'iv_{dte_long}d')

    df = df.merge(short_df, on=['trade_date', 'underlying_symbol'], how='left')
    df = df.merge(long_df,  on=['trade_date', 'underlying_symbol'], how='left')
    df['vol_term_structure'] = df[f'iv_{dte_long}d'] - df[f'iv_{dte_short}d']

    # 2) ATM-IV: pick nearest among ATM subset
    atm_df = df[df['moneyness'].between(1 - atm_tol, 1 + atm_tol)]
    atm_short = pick_nearest_iv(atm_df, dte_short, f'atm_iv_{dte_short}d')
    atm_long  = pick_nearest_iv(atm_df, dte_long,  f'atm_iv_{dte_long}d')

    df = df.merge(atm_short, on=['trade_date', 'underlying_symbol'], how='left')
    df = df.merge(atm_long,  on=['trade_date', 'underlying_symbol'], how='left')
    df['calendar_skew'] = df[f'atm_iv_{dte_long}d'] - df[f'atm_iv_{dte_short}d']

    return df

def add_surface_skew(
    df: pd.DataFrame,
    delta_target: float = 0.25,
    delta_tol: float = 0.01
) -> pd.DataFrame:
    """
    Add surface skew feature: difference between call and put IV at a given delta magnitude.
    - iv_call_{target*100} and iv_put_{target*100}: average implied_volatility for calls/puts with |delta| ≈ target
    - surface_skew_{target*100} = iv_call - iv_put
    """
    # 1) identify 25Δ calls and puts
    call_mask = (
        (df['option_type'] == 'call') &
        df['delta'].between(delta_target - delta_tol, delta_target + delta_tol)
    )
    put_mask = (
        (df['option_type'] == 'put') &
        df['delta'].between(-delta_target - delta_tol, -delta_target + delta_tol)
    )

    iv_call = (
        df[call_mask]
        .groupby(['trade_date', 'underlying_symbol'])['implied_volatility']
        .mean()
        .rename(f'iv_{int(delta_target*100)}c')
        .reset_index()
    )
    iv_put = (
        df[put_mask]
        .groupby(['trade_date', 'underlying_symbol'])['implied_volatility']
        .mean()
        .rename(f'iv_{int(delta_target*100)}p')
        .reset_index()
    )

    # 2) merge and compute surface skew
    skew_df = iv_call.merge(iv_put, on=['trade_date', 'underlying_symbol'], how='outer')
    skew_df[f'surface_skew_{int(delta_target*100)}'] = (
        skew_df[f'iv_{int(delta_target*100)}c'] - skew_df[f'iv_{int(delta_target*100)}p']
    )

    # 3) merge back to full DataFrame
    df = df.merge(
        skew_df[['trade_date', 'underlying_symbol', f'surface_skew_{int(delta_target*100)}']],
        on=['trade_date', 'underlying_symbol'],
        how='left'
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_cross_sectional_features(
    df: pd.DataFrame,
    rank_cols: list = None,
    zscore_cols: list = None
) -> pd.DataFrame:
    """
    Add cross-sectional rank and z-score features for each date:
    - For each col in rank_cols, compute percentile rank within that date.
    - For each col in zscore_cols, compute (value - mean)/std within that date.
    
    Assumes df has a 'date' column.
    """
    df_out = df.copy()
    
    # Default columns if not provided
    rank_cols = rank_cols or ['implied_volatility']
    zscore_cols = zscore_cols or ['realized_vol_10d']
    
    # Compute percentile ranks
    for col in rank_cols:
        rank_name = f"{col}_rank"
        df_out[rank_name] = (
            df_out
            .groupby('trade_date')[col]
            .rank(pct=True)
        )
    
    # Compute z-scores
    for col in zscore_cols:
        z_name = f"{col}_zscore"
        df_out[z_name] = (
            df_out
            .groupby('trade_date')[col]
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
        )
    
    return df_out

def add_cross_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-feature interaction columns:
      - moneyness_x_rsi: moneyness * rsi_14
      - gamma_over_theta: gamma / |theta|
      - delta_x_vix_regime: delta * regime (1 if vix_close above median, else 0)
      - delta_gamma_interaction: delta * gamma
    """
    df = df.copy()
    # 1) moneyness × RSI
    df['moneyness_x_rsi'] = df['moneyness'] * df['rsi_14']
    
    # 2) gamma / |theta|
    df['gamma_over_theta'] = df['gamma'] / (df['theta'].abs() + 1e-9)
    
    # 3) delta × VIX regime
    median_vix = df['vix_close'].median()
    df['vix_regime'] = (df['vix_close'] > median_vix).astype(int)
    df['delta_x_vix_regime'] = df['delta'] * df['vix_regime']
    df['delta_x_moneyness']     = df['delta'] * df['moneyness']
    df['delta_iv_interaction']  = df['delta'] * df['implied_volatility']
    df['moneyness_sqrt_dte']    = df['moneyness'] * np.sqrt(df['dte_days'])
    
    # 4) delta × gamma
    df['delta_gamma_interaction'] = df['delta'] * df['gamma']
    
    return df

def add_advanced_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["vega_x_iv_rank"] = df["vega"] * df["implied_volatility_rank"]
    df["rsi_x_vol_chg"] = df["rsi_14"] * df["stk_vol_chg_pct"]
    df["price_mom_x_vol"] = df["price_change_3d"] * df["opt_volume_rank"]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING/LAG FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["underlying_symbol", "trade_date"]).copy()
    lag_cols = ["stk_return_1d", "price_change_3d", "rsi_14"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("underlying_symbol")[col].shift(1)
    return df

def add_rolling_vol_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["underlying_symbol", "trade_date"]).copy()
    df["realized_vol_20d"] = (
        df.groupby("underlying_symbol")["stk_return_1d"]
        .rolling(20)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    df["vol_ratio_10d_20d"] = df["realized_vol_10d"] / (df["realized_vol_20d"] + 1e-9)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STOCK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["underlying_symbol", "trade_date"]).copy()

    # MACD diff
    macd = ta.trend.MACD(close=df["stk_close"])
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Band width
    bb = ta.volatility.BollingerBands(close=df["stk_close"])
    df["bollinger_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
    df["percent_b"] = bb.bollinger_pband()

    # ADX
    adx = ta.trend.ADXIndicator(high=df["stk_high"], low=df["stk_low"], close=df["stk_close"])
    df["adx"] = adx.adx()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df["stk_high"], low=df["stk_low"], close=df["stk_close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    return df

# ─────────────────────────────────────────────────────────────────────────────
# BUCKET FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_categorical_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moneyness buckets
    def bucket_moneyness(m):
        if m < 0.97:
            return "OTM"
        elif m > 1.03:
            return "ITM"
        else:
            return "ATM"
    df["moneyness_bucket"] = df["moneyness"].apply(bucket_moneyness)

    # DTE buckets
    def bucket_dte(d):
        if d <= 14:
            return "short"
        elif d <= 30:
            return "medium"
        else:
            return "long"
    df["dte_bucket"] = df["dte_days"].apply(bucket_dte)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# LOADING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def load_ml_features() -> pd.DataFrame:
    logging.info("Loading stock and options data from DB…")
    # Refresh the materialized view
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW mv_ml_features"))

    # Pull everything
    sql = """
    SELECT *
      FROM mv_ml_features
     WHERE moneyness BETWEEN 0.95 AND 1.05
       AND dte_days  BETWEEN 7    AND 45
     ORDER BY trade_date, ticker, option_symbol
    """

    df = pd.read_sql(
        text(sql),
        con=engine,
        parse_dates=["trade_date"]  # still converts trade_date into a datetime
    )
    df = df.rename(columns={"opt_close": "mid_price", "ticker": "underlying_symbol"})
    logging.info(f"  ▶ Loaded {len(df)} data rows from db")
    return df

def filter_option_features(df_data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Filtering stock and options data for ML model…")
    df = df_data.copy()
    # ensure date column is datetime
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # compute dynamic window for macro data
    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()

    # start 60 days before earliest feature date
    macro_start = (min_date - pd.Timedelta(days=60)).date().isoformat()
    # end just after the latest feature date
    macro_end   = (max_date + pd.Timedelta(days=1)).date().isoformat()

    macro_df = fetch_macro_df(start_date=macro_start, end_date=macro_end)
    logging.info(f"▶ Fetching macro data from {macro_start} to {macro_end}")

    df = add_categorical_buckets(df)
    df = add_surface_skew(df)
    df = add_vol_term_structure_and_calendar_skew(df)
    df = add_cross_sectional_features(
        df,
        rank_cols=[
            'implied_volatility',
            'stk_volume',
            'opt_volume',
            'delta',
            'vega',
            'surface_skew_25',
            'calendar_skew'
        ],
        zscore_cols=[
            'realized_vol_10d',
            'vol_term_structure',
            'calendar_skew',
            'stk_return_1d',
            'price_change_3d',
            'stk_range_pct',
            'ma5_ma10_diff',
            'rsi_14'
        ]
    )
    df = add_cross_feature_interactions(df)
    df = add_lag_features(df)
    df = add_rolling_vol_ratio(df)
    df = add_advanced_interactions(df)
    df = add_technical_indicators(df)
    df = add_macro_features(df, macro_df)

    logging.info(f"  ▶ Final Feature DataFrame shape (with ATM skew): {df.shape}")
    return df

def load_feature_dataframe() -> pd.DataFrame:
    """
    Load raw data from DB, compute ML features, and build forward-return targets.

    Args:
        mode: "next_day" or "max_3day" for target construction.
    Returns:
        df_ready with all feature columns, 'target_return', 'hold_days', and 'target_label'.
    """
    # Step 1: load from database
    df_raw = load_ml_features()
    logging.info(f"Loaded raw data: {len(df_raw)} rows")

    # Step 2: compute ML features
    df_feat = filter_option_features(df_raw)
    logging.info(f"Computed features: {df_feat.shape[1]} columns")

    return df_feat