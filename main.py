"""
Entry point — wires all components together and runs the bot.

Usage:
  python main.py            ← live trading on all enabled symbols
  python main.py --watch    ← watch-only: display signals without placing trades
"""

import asyncio
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.data.tick_store import TickStore
from src.strategies.rsi_reversal import RSIReversalStrategy
from src.risk.manager import RiskManager
from src.execution.trader import Trader
from src.monitoring.dashboard import Dashboard
from src.monitoring.telegram_alerts import build_alerter


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
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Per-symbol context ─────────────────────────────────────────────────

@dataclass
class SymbolBot:
    symbol: str
    tick_store: TickStore
    strategy: RSIReversalStrategy
    risk: RiskManager
    trader: Trader


def build_symbol_config(symbol_entry: dict, base_strategy: dict) -> dict:
    """Merge symbol-level overrides onto the base strategy config."""
    cfg = dict(base_strategy)
    for k, v in symbol_entry.items():
        if k not in ("symbol", "enabled"):
            cfg[k] = v
    cfg["symbol"] = symbol_entry["symbol"]
    return cfg


# ── Midnight daily reset ────────────────────────────────────────────────

async def midnight_reset_loop(bots: list[SymbolBot], alerter):
    """Fires once per day at midnight: resets daily P&L baselines and sends summaries."""
    while True:
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0
        )
        await asyncio.sleep((next_midnight - now).total_seconds())

        for bot in bots:
            r = bot.risk
            logger.info(
                f"Daily summary [{bot.symbol}] | "
                f"Trades: {r.total_trades} | W/L: {r.wins}/{r.losses} | "
                f"WR: {r.win_rate}% | Net: {r.net_profit:+.2f} | Bal: {r.current_balance:.2f}"
            )
            if alerter:
                await alerter.send_daily_summary(
                    symbol=bot.symbol,
                    trades=r.total_trades,
                    wins=r.wins,
                    losses=r.losses,
                    net_pnl=r.net_profit,
                    win_rate=r.win_rate,
                    balance=r.current_balance,
                )
            r.reset_daily()


# ── Main bot loop ───────────────────────────────────────────────────────

async def run(watch_only: bool = False):
    config       = load_config()
    base_strategy = config["strategy"]
    risk_cfg     = config["risk"]

    api_token = os.getenv("DERIV_API_TOKEN")
    _raw_app_id = os.getenv("DERIV_APP_ID", "1089")
    app_id = _raw_app_id if _raw_app_id.isdigit() else "1089"

    if not api_token:
        logger.error("DERIV_API_TOKEN not set in .env")
        return

    # ── Telegram ─────────────────────────────────────────────────
    alerter = build_alerter(config)
    if alerter:
        logger.info("Telegram alerts enabled")

    # ── Connect ──────────────────────────────────────────────────
    client = DerivClient(api_token=api_token, app_id=app_id)
    await client.connect()

    total_balance = await client.get_balance()
    logger.info(f"Account balance: {total_balance:.2f}")

    # ── Build per-symbol bots ─────────────────────────────────────
    symbol_entries = config.get("symbols") or [{"symbol": "R_25", "enabled": True}]
    active_entries = [e for e in symbol_entries if e.get("enabled", True)]

    if not active_entries:
        logger.error("No symbols enabled in config.yaml")
        return

    num_symbols    = len(active_entries)
    balance_per    = total_balance / num_symbols
    logger.info(f"Trading {num_symbols} symbol(s) | {balance_per:.2f} balance allocated each")

    bots: list[SymbolBot] = []
    for entry in active_entries:
        sym_cfg  = build_symbol_config(entry, base_strategy)
        symbol   = sym_cfg["symbol"]

        tick_store = TickStore(rsi_period=sym_cfg["rsi_period"])
        strategy   = RSIReversalStrategy(sym_cfg)
        risk       = RiskManager(risk_cfg, starting_balance=balance_per)
        trader     = Trader(
            client, risk, symbol,
            sym_cfg["contract_duration"],
            sym_cfg["contract_duration_unit"],
            strategy=strategy,
            alerter=alerter,
        )

        # Register per-symbol tick handler
        async def make_handler(b: SymbolBot, wonly: bool):
            async def handler(tick: dict):
                b.tick_store.add(tick)
                signal = b.strategy.evaluate(b.tick_store)
                if not wonly and signal.action != "HOLD":
                    await b.trader.execute(signal)
            return handler

        bot = SymbolBot(symbol=symbol, tick_store=tick_store,
                        strategy=strategy, risk=risk, trader=trader)
        bots.append(bot)
        client.on_tick(symbol, await make_handler(bot, watch_only))

    # ── Dashboard (tracks first symbol for display) ───────────────
    primary = bots[0]
    dashboard = Dashboard(
        primary.tick_store,
        primary.risk,
        ema_period=base_strategy.get("ema_trend_period", 0),
    )

    mode_label = "WATCH ONLY" if watch_only else "LIVE TRADING"
    symbols_label = ", ".join(b.symbol for b in bots)
    logger.info(f"Bot started | Mode: {mode_label} | Symbols: {symbols_label}")

    if alerter:
        await alerter.send_message(
            f"🤖 *Bot started*\nMode: {mode_label}\nSymbols: `{symbols_label}`\nBalance: `{total_balance:.2f} USD`"
        )

    dashboard.start()

    # ── Subscribe ticks for all symbols ──────────────────────────
    for bot in bots:
        await client.subscribe_ticks(bot.symbol)

    # ── Start midnight reset loop ─────────────────────────────────
    asyncio.create_task(midnight_reset_loop(bots, alerter))

    # ── Keep running ─────────────────────────────────────────────
    try:
        while True:
            dashboard.refresh()
            all_halted = all(b.risk.is_halted for b in bots)
            if all_halted:
                logger.warning("All symbols halted. Restart to resume.")
                break
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Bot stopped by user")
    finally:
        dashboard.stop()
        await client.disconnect()

        print("\nSession summary:")
        for bot in bots:
            r = bot.risk
            print(f"  [{bot.symbol}] Trades: {r.total_trades} | "
                  f"W/L: {r.wins}/{r.losses} | WR: {r.win_rate}% | "
                  f"Net: {r.net_profit:+.2f} | Bal: {r.current_balance:.2f}")

        if alerter:
            for bot in bots:
                r = bot.risk
                await alerter.send_daily_summary(
                    symbol=bot.symbol,
                    trades=r.total_trades,
                    wins=r.wins,
                    losses=r.losses,
                    net_pnl=r.net_profit,
                    win_rate=r.win_rate,
                    balance=r.current_balance,
                )


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
