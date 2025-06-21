from pathlib import Path
import os
import logging
import requests
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import pandas_market_calendars as mcal

from concurrent.futures import ThreadPoolExecutor, as_completed

from api.signal_generator.models import load_models, TradeSignal
from api.signal_generator.features import build_feature_rows
from api.signal_generator.fetchers import fetch_tlt_return_5d, fetch_trending_tickers
from api.signal_generator.config import TOP_TICKERS, MIN_PRED_RETURN, FEATURE_COLS, TRADE_MANAGER_URL
from api.utils import is_market_open, get_last_n_trading_days

# Load pre-trained models (mean, quantile, ranker, meta)
MODELS = load_models()

def process_ticker(ticker: str, tlt_return_5d: float, as_of_date: dt.date, lookback_start: dt.date) -> list[dict]:
    try:
        logging.info(f"Processing {ticker} as-of {as_of_date}...")
        # Build features using explicit date bounds
        rows = build_feature_rows(
            ticker,
            tlt_return_5d,
            as_of_date=as_of_date,
            lookback_start=lookback_start,
        )
        if not rows:
            return []

        df = pd.DataFrame([{k: r[k] for k in FEATURE_COLS} for r in rows])

        pred_mean = MODELS["mean_model"].predict(df)
        pred_q10 = MODELS["quantile_models"][0.1].predict(df)
        pred_q50 = MODELS["quantile_models"][0.5].predict(df)
        pred_q90 = MODELS["quantile_models"][0.9].predict(df)
        pred_rank = MODELS["ranker_model"].predict(df)

        meta_input = np.column_stack([pred_mean, pred_q10, pred_q50, pred_q90, pred_rank])
        pred_meta = MODELS["meta_model"].predict(meta_input)

        for row, pred in zip(rows, pred_meta):
            row["pred_return"] = pred

        return rows

    except Exception as e:
        logging.error(f"Failed to process {ticker}: {e}")
        return []


def send_signal_to_trade_manager(signal: TradeSignal):
    try:
        url = f"{TRADE_MANAGER_URL}/trade/signal"
        payload = signal.dict()
        if isinstance(payload.get("expiry"), dt.date):
            payload["expiry"] = payload["expiry"].isoformat()

        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            logging.warning(f"TradeManager returned {resp.status_code}: {resp.text}")
        else:
            logging.info(f"✅ Sent signal for {signal.option_symbol}")

    except Exception as e:
        logging.error(f"Failed to send signal to TradeManager: {e}")


def run_signal_generation():
    try:
        logging.info("Starting signal generation")

        # Determine enterprise 'as-of' date and lookback window
        tz = pytz.timezone("US/Eastern")
        today = dt.datetime.now(tz).date()
        [as_of_date] = get_last_n_trading_days(end_date=today, n=1)
        lookback_days = 5
        last_lookback = get_last_n_trading_days(end_date=as_of_date + dt.timedelta(days=1), n=lookback_days)
        lookback_start = last_lookback[0]
        logging.info(f"Using data from {lookback_start} to {as_of_date} for features")

        # Fetch shared TLT return (5-day)
        tlt_return_5d = fetch_tlt_return_5d()
        logging.info(f"TLT {lookback_days}-day return: {tlt_return_5d:.4f}")

        # Build universe
        results = []
        trending_tickers = fetch_trending_tickers(top_n=5)
        full_universe = list(set(TOP_TICKERS + trending_tickers))
        logging.info(f"Generating signals for {len(full_universe)} tickers: {trending_tickers}")

        # Process in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_ticker, ticker, tlt_return_5d, as_of_date, lookback_start): ticker
                for ticker in full_universe
            }
            for future in as_completed(futures):
                results.extend(future.result())

        # Filter duplicates by expiry
        filtered = {}
        for row in sorted(results, key=lambda x: -x["pred_return"]):
            key = (row["ticker"], row["expiry"])
            if key not in filtered:
                filtered[key] = row
        filtered_signals = list(filtered.values())

        # Log top 10 pred_return of filtered_signals
        if filtered_signals:
            sorted_filtered = sorted(filtered_signals, key=lambda x: -x.get("pred_return", 0))
            top10_filtered = sorted_filtered[:10]
            logging.info("Top 10 pred_return after deduplication:")
            for r in top10_filtered:
                logging.info(f"  {r['ticker']} {r['expiry']} -> {r['pred_return']:.4f}")

        # Quantile threshold
        q10_df = pd.DataFrame([{k: r[k] for k in FEATURE_COLS} for r in filtered_signals])
        if q10_df.empty:
            logging.warning(f"No feature data for {as_of_date}. Skipping signal generation.")
            return
        q10_preds = MODELS["quantile_models"][0.1].predict(q10_df)
        for row, q10 in zip(filtered_signals, q10_preds):
            row["q10_pred"] = q10

        final_signals = [
            r for r in filtered_signals
            if r["pred_return"] > r["q10_pred"] and r["pred_return"] >= MIN_PRED_RETURN
        ]
        if not final_signals:
            logging.info("No signals passed filtering.")
            return

        final_df = (
            pd.DataFrame(final_signals)
            .sort_values(by='pred_return', ascending=False)
            .groupby('ticker')
            .head(2)
        )

        sent = []
        for rec in final_df.to_dict("records"):
            try:
                opt = rec["option_symbol"]
                if opt.startswith("O:"):
                    opt = opt[2:]
                expiry = pd.to_datetime(rec["expiry"]).date()
                signal = TradeSignal(
                    ticker=rec["ticker"],
                    option_symbol=opt,
                    option_price=rec["option_price"],
                    strike=rec["strike"],
                    expiry=expiry,
                    pred_return=rec["pred_return"]
                )
                send_signal_to_trade_manager(signal)
                sent.append(rec)
            except Exception as e:
                logging.error(f"Failed to process/send signal: {e}")

        if sent:
            debug_df = pd.DataFrame(sent)
            out_dir = Path(__file__).parent / "data" / "sent_signals"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_df.to_csv(out_dir / f"sent_signals_debug_{stamp}.csv", index=False)
            debug_df.to_csv(out_dir / "sent_signals_latest.csv", index=False)
            logging.info(f"Wrote {len(sent)} signals to sent_signals_latest.csv")

    except Exception as e:
        logging.error(f"Signal generation failed: {e}")
