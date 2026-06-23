"""
Backtest runner.

Usage:
  python backtest.py                        # single window, 5000 ticks
  python backtest.py --count 10000          # more ticks = more reliable stats
  python backtest.py --fresh                # re-fetch ticks from Deriv (ignore cache)
  python backtest.py --balance 500          # simulate with $500 starting balance
  python backtest.py --payout 0.95          # override payout % (check your Deriv account)
  python backtest.py --no-ema               # disable EMA trend filter to compare
  python backtest.py --no-bb                # disable Bollinger Band filter to compare
  python backtest.py --windows 5            # validate across 5 independent non-overlapping windows
  python backtest.py --windows 5 --count 5000   # fetch 25k ticks, split into 5 x 5k windows

Multi-window mode:
  Fetches (count * windows) ticks total, splits them chronologically into N
  non-overlapping windows, runs an independent backtest on each window, and
  prints a per-window breakdown plus an aggregate summary.  Params that are
  profitable in EVERY window are genuinely robust.

Tick data is cached to data/<symbol>_<total_count>.json.
"""

import argparse
import copy
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.data.history import fetch_ticks
from src.backtest.engine import BacktestEngine
from src.backtest.report import print_report, print_multi_window_report


load_dotenv()

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Deriv bot backtester")
    parser.add_argument("--count",   type=int,   default=5000,   help="Ticks per window (default: 5000)")
    parser.add_argument("--balance", type=float, default=1000.0,  help="Simulated starting balance per window (default: 1000)")
    parser.add_argument("--payout",  type=float, default=0.87,    help="Payout %% as decimal e.g. 0.87 = 87%% (default: 0.87)")
    parser.add_argument("--fresh",   action="store_true",          help="Re-fetch ticks from Deriv even if cached")
    parser.add_argument("--no-ema",  action="store_true",          help="Disable EMA trend filter")
    parser.add_argument("--no-bb",   action="store_true",          help="Disable Bollinger Band filter")
    parser.add_argument("--no-atr-filter", action="store_true",    help="Disable ATR ranging-market gate")
    parser.add_argument("--windows", type=int,   default=1,        help="Number of independent windows to validate across (default: 1)")
    parser.add_argument("--symbol",  type=str,   default=None,     help="Symbol to backtest (e.g. R_75). Overrides config.")
    args = parser.parse_args()

    config = load_config()
    strategy_cfg = copy.deepcopy(config["strategy"])
    risk_cfg     = copy.deepcopy(config["risk"])

    # Apply CLI overrides
    if args.no_ema:
        strategy_cfg["ema_trend_period"] = 0
    if args.no_bb:
        strategy_cfg["use_bb_filter"] = False
    if args.no_atr_filter:
        strategy_cfg["use_atr_filter"] = False

    app_id    = os.getenv("DERIV_APP_ID", "1089")
    api_token = os.getenv("DERIV_API_TOKEN")

    # --symbol overrides: merge that symbol's config on top of base strategy
    if args.symbol:
        target = args.symbol.upper()
        strategy_cfg["symbol"] = target
        # apply any per-symbol overrides from the symbols list
        for entry in config.get("symbols", []):
            if entry["symbol"] == target:
                for k, v in entry.items():
                    if k not in ("symbol", "enabled"):
                        strategy_cfg[k] = v
                break
    elif "symbol" not in strategy_cfg:
        symbols_list = config.get("symbols", [])
        enabled = [s for s in symbols_list if s.get("enabled", True)]
        strategy_cfg["symbol"] = enabled[0]["symbol"] if enabled else "R_25"

    symbol = strategy_cfg["symbol"]

    n_windows   = max(1, args.windows)
    total_ticks = args.count * n_windows

    # ── Fetch ticks ──────────────────────────────────────────────
    print(f"Fetching {total_ticks:,} ticks for {n_windows} window(s) of {args.count:,} each...")
    try:
        ticks = fetch_ticks(symbol, count=total_ticks, app_id=app_id, api_token=api_token, fresh=args.fresh)
    except Exception as e:
        logger.error(f"Failed to fetch tick history: {e}")
        sys.exit(1)

    if len(ticks) < 100:
        logger.error(f"Only {len(ticks)} ticks returned — need at least 100 to run.")
        sys.exit(1)

    # ── Single-window mode ───────────────────────────────────────
    if n_windows == 1:
        engine = BacktestEngine(strategy_cfg=strategy_cfg, risk_cfg=risk_cfg, payout_pct=args.payout)
        result = engine.run(ticks, starting_balance=args.balance)
        print_report(result, strategy_cfg, risk_cfg)
        return

    # ── Multi-window mode ────────────────────────────────────────
    # Split chronologically into N equal non-overlapping slices.
    window_size = len(ticks) // n_windows
    results = []

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        window_ticks = ticks[start:end]
        print(f"  Window {i+1}/{n_windows}: ticks {start}–{end-1} ({len(window_ticks):,} ticks)")

        engine = BacktestEngine(strategy_cfg=strategy_cfg, risk_cfg=risk_cfg, payout_pct=args.payout)
        result = engine.run(window_ticks, starting_balance=args.balance)
        results.append(result)

    print_multi_window_report(results, strategy_cfg, risk_cfg)


if __name__ == "__main__":
    main()
