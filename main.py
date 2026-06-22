"""
Entry point — wires all components together and runs the bot.

Usage:
  python main.py            ← run with live trading (demo account)
  python main.py --watch    ← watch-only mode: shows prices/RSI, no trades placed
"""

import asyncio
import argparse
import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.data.tick_store import TickStore
from src.strategies.rsi_reversal import RSIReversalStrategy
from src.risk.manager import RiskManager
from src.execution.trader import Trader
from src.monitoring.dashboard import Dashboard


# ── Setup ──────────────────────────────────────────────────────────────

load_dotenv()
os.makedirs("logs", exist_ok=True)

logger.remove()
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}",
)
# Keep errors visible in terminal even when dashboard is running
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Main bot loop ───────────────────────────────────────────────────────

async def run(watch_only: bool = False):
    config = load_config()
    strategy_cfg = config["strategy"]
    risk_cfg = config["risk"]

    api_token = os.getenv("DERIV_API_TOKEN")
    app_id = os.getenv("DERIV_APP_ID", "1089")

    if not api_token:
        logger.error("DERIV_API_TOKEN not set in .env")
        return

    symbol = strategy_cfg["symbol"]
    rsi_period = strategy_cfg["rsi_period"]
    duration = strategy_cfg["contract_duration"]
    duration_unit = strategy_cfg["contract_duration_unit"]

    # ── Connect ──────────────────────────────────────────────────
    client = DerivClient(api_token=api_token, app_id=app_id)
    await client.connect()

    starting_balance = await client.get_balance()
    logger.info(f"Starting balance: {starting_balance}")

    # ── Initialise components ────────────────────────────────────
    tick_store = TickStore(rsi_period=rsi_period)
    strategy = RSIReversalStrategy(strategy_cfg)
    risk = RiskManager(risk_cfg, starting_balance=starting_balance)
    trader = Trader(client, risk, symbol, duration, duration_unit, strategy=strategy)
    dashboard = Dashboard(tick_store, risk, ema_period=strategy_cfg.get("ema_trend_period", 50))

    mode_label = "WATCH ONLY" if watch_only else "LIVE TRADING"
    logger.info(f"Bot started | Mode: {mode_label} | Symbol: {symbol}")

    dashboard.start()

    # ── Tick handler ─────────────────────────────────────────────
    async def on_tick(tick: dict):
        tick_store.add(tick)
        signal = strategy.evaluate(tick_store)
        dashboard.update_signal(signal)
        dashboard.refresh()

        if not watch_only and signal.action != "HOLD":
            await trader.execute(signal)

    client.on_tick(on_tick)
    await client.subscribe_ticks(symbol)

    # ── Keep running until interrupted ──────────────────────────
    try:
        while True:
            if risk.is_halted:
                logger.warning("Bot halted by risk manager. Restart to reset daily limits.")
                break
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Bot stopped by user")
    finally:
        dashboard.stop()
        await client.disconnect()
        print(f"\nSession summary:")
        print(f"  Trades : {risk.total_trades}")
        print(f"  Wins   : {risk.wins}  |  Losses: {risk.losses}")
        print(f"  Win %  : {risk.win_rate}%")
        print(f"  Net P&L: {risk.net_profit:+.2f}")
        print(f"  Balance: {risk.current_balance:.2f}")


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deriv Indices Trading Bot")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch-only mode: display prices and signals without placing trades",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(watch_only=args.watch))
    except KeyboardInterrupt:
        pass
