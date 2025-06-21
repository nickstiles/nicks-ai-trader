import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure backend/python is on sys.path so imports work when running from project root
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from models.backtest_options import run_backtest_credit_spreads


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep confidence thresholds for the overnight credit‐spread backtester."
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL",
        help="Ticker symbol to backtest (default: AAPL)"
    )
    parser.add_argument(
        "--period", type=str, default="6mo",
        help="Data period for backtest (e.g. '6mo', '1y') (default: 6mo)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d",
        help="Data interval for backtest (only '1d' for options) (default: 1d)"
    )
    parser.add_argument(
        "--expiry-offset", type=int, default=7,
        help="Days until option expiration (default: 7)"
    )
    parser.add_argument(
        "--quantity", type=int, default=1,
        help="Number of contracts to trade per signal (default: 1)"
    )
    parser.add_argument(
        "--threshold-min", type=float, default=0.50,
        help="Minimum confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--threshold-max", type=float, default=0.80,
        help="Maximum confidence threshold (default: 0.80)"
    )
    parser.add_argument(
        "--threshold-step", type=float, default=0.05,
        help="Step size for confidence threshold (default: 0.05)"
    )
    parser.add_argument(
        "--commission", type=float, default=0.65,
        help="Commission per contract (default: 0.65)"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.05,
        help="Slippage per contract (default: 0.05)"
    )
    parser.add_argument(
        "--iv-rank-threshold", type=float, default=0.70,
        help="Maximum IV‐Rank to allow selling (default: 0.70)"
    )
    parser.add_argument(
        "--width1", type=float, default=2.0,
        help="Offset (in $) from ATM for short leg (default: 2.0)"
    )
    parser.add_argument(
        "--width2", type=float, default=5.0,
        help="Offset (in $) from ATM for long leg (default: 5.0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + 1e-9,
        args.threshold_step
    )
    results = []

    for thresh in thresholds:
        print(f"Running credit‐spread backtest @ confidence ≥ {thresh:.2f} …")
        try:
            res = run_backtest_credit_spreads(
                ticker=args.ticker,
                period=args.period,
                timeframe=args.timeframe,
                expiry_offset=args.expiry_offset,
                quantity=args.quantity,
                confidence_threshold_up=thresh,
                confidence_threshold_down=thresh,
                commission_per_contract=args.commission,
                slippage_per_contract=args.slippage,
                iv_rank_threshold=args.iv_rank_threshold,
                width1=args.width1,
                width2=args.width2
            )
        except Exception as e:
            print(f"  [ERROR] threshold={thresh:.2f} → {e}")
            continue

        results.append({
            "threshold":       round(thresh, 2),
            "num_trades":      res["num_trades"],
            "net_return":      res["net_return"],
            "win_rate":        res["win_rate"],
            "sharpe_ratio":    res["sharpe_ratio"],
            "max_drawdown":    res["max_drawdown"]
        })

    if results:
        df_results = pd.DataFrame(results)
        df_results["net_return_%"]   = (df_results["net_return"] * 100).round(2)
        df_results["win_rate_%"]     = (df_results["win_rate"] * 100).round(2)
        df_results["max_drawdown_%"] = (df_results["max_drawdown"] * 100).round(2)

        display_cols = [
            "threshold",
            "num_trades",
            "net_return_%",
            "win_rate_%",
            "sharpe_ratio",
            "max_drawdown_%"
        ]

        print("\nCredit‐Spread Backtest Sweep Results:")
        print(df_results[display_cols].to_string(index=False))
    else:
        print("No valid backtest results were generated.")


if __name__ == "__main__":
    main()