#!/usr/bin/env python3
"""
sweep_thresholds.py

Run a backtest for a range of confidence thresholds and report performance metrics.
"""

import argparse
import numpy as np
import pandas as pd

from data.ingestion import fetch_ohlcv
from models.backtest import run_backtest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep confidence thresholds and report backtest metrics."
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL",
        help="Ticker symbol to backtest (default: AAPL)"
    )
    parser.add_argument(
        "--period", type=str, default="30d",
        help="Data period for backtest, e.g. '30d', '3mo' (default: 30d)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="5m",
        help="Data interval for backtest, e.g. '5m', '1h', '1d' (default: 5m)"
    )
    parser.add_argument(
        "--quantity", type=int, default=10,
        help="Number of shares/contracts to trade per signal (default: 10)"
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.50,
        help="Minimum confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.90,
        help="Maximum confidence threshold (default: 0.90)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.05,
        help="Step size for confidence threshold (default: 0.05)"
    )
    parser.add_argument(
        "--commission", type=float, default=0.005,
        help="Commission per share (default: $0.005)"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0005,
        help="Slippage percentage (default: 0.05%)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fetch OHLCV data once
    print(f"Fetching {args.ticker} data: period={args.period}, interval={args.timeframe}…")
    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.timeframe)
    if df.empty:
        print("Error: No market data returned. Check ticker, period, or timeframe.")
        return

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    results = []

    for thresh in thresholds:
        print(f"Running backtest at confidence threshold = {thresh:.2f} …")
        try:
            res = run_backtest(
                df,
                initial_capital=100_000.0,
                quantity=args.quantity,
                confidence_threshold=thresh,
                commission_per_share=args.commission,
                slippage_pct=args.slippage
            )
        except Exception as e:
            print(f"  [ERROR] Backtest failed at threshold={thresh:.2f}: {e}")
            continue

        results.append({
            "threshold":    round(thresh, 2),
            "net_return":   res["net_return"],   # <-- changed from "total_return"
            "win_rate":     res["win_rate"],
            "num_trades":   res["num_trades"],
            "sharpe_ratio": res["sharpe_ratio"],
            "max_drawdown": res["max_drawdown"]
        })

    # Build a DataFrame and display
    if results:
        df_results = pd.DataFrame(results)
        # Convert to more readable percentages
        df_results["net_return_%"]   = (df_results["net_return"] * 100).round(2)
        df_results["win_rate_%"]     = (df_results["win_rate"] * 100).round(2)
        df_results["max_drawdown_%"] = (df_results["max_drawdown"] * 100).round(2)

        columns = [
            "threshold",
            "num_trades",
            "net_return_%",
            "win_rate_%",
            "sharpe_ratio",
            "max_drawdown_%"
        ]

        print("\nBacktest Sweep Results:")
        print(df_results[columns].to_string(index=False))
    else:
        print("No valid results were generated.")


if __name__ == "__main__":
    main()
