"""
Backtest runner.

Usage:
  python backtest.py                        # use defaults from config.yaml
  python backtest.py --count 10000          # more ticks = more reliable stats
  python backtest.py --fresh                # re-fetch ticks from Deriv (ignore cache)
  python backtest.py --balance 500          # simulate with $500 starting balance
  python backtest.py --payout 0.95          # override payout % (check your Deriv account)
  python backtest.py --no-ema               # disable EMA trend filter to compare
  python backtest.py --no-bb                # disable Bollinger Band filter to compare

The first run downloads ticks and caches them to data/<symbol>_<count>.json.
Subsequent runs load from cache instantly.
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
from src.backtest.report import print_report


load_dotenv()

logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Deriv bot backtester")
    parser.add_argument("--count",   type=int,   default=5000,  help="Number of historical ticks to test (default: 5000)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Simulated starting balance (default: 1000)")
    parser.add_argument("--payout",  type=float, default=0.87,   help="Payout %% as decimal e.g. 0.87 = 87%% (default: 0.87)")
    parser.add_argument("--fresh",   action="store_true",         help="Re-fetch ticks from Deriv even if cached")
    parser.add_argument("--no-ema",  action="store_true",         help="Disable EMA trend filter")
    parser.add_argument("--no-bb",   action="store_true",         help="Disable Bollinger Band filter")
    args = parser.parse_args()

    config = load_config()
    strategy_cfg = copy.deepcopy(config["strategy"])
    risk_cfg     = copy.deepcopy(config["risk"])

    # Apply CLI overrides
    if args.no_ema:
        strategy_cfg["ema_trend_period"] = 0
    if args.no_bb:
        strategy_cfg["use_bb_filter"] = False

    app_id    = os.getenv("DERIV_APP_ID", "1089")
    api_token = os.getenv("DERIV_API_TOKEN")
    symbol    = strategy_cfg["symbol"]

    # ── Fetch ticks ──────────────────────────────────────────────
    try:
        ticks = fetch_ticks(symbol, count=args.count, app_id=app_id, api_token=api_token, fresh=args.fresh)
    except Exception as e:
        logger.error(f"Failed to fetch tick history: {e}")
        sys.exit(1)

    if len(ticks) < 100:
        logger.error(f"Only {len(ticks)} ticks returned — need at least 100 to run.")
        sys.exit(1)

    # ── Run backtest ─────────────────────────────────────────────
    engine = BacktestEngine(
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
        payout_pct=args.payout,
    )
    result = engine.run(ticks, starting_balance=args.balance)

    # ── Print results ────────────────────────────────────────────
    print_report(result, strategy_cfg, risk_cfg)


if __name__ == "__main__":
    main()
