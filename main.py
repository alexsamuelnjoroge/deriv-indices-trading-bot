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
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.client import DerivClient
from src.data.tick_store import TickStore
from src.data.history import fetch_candles_async
from src.strategies.base import BaseStrategy
from src.strategies.rsi_reversal import RSIReversalStrategy
from src.strategies.gold_trend import GoldTrendStrategy
from src.strategies.macd_trend import MACDTrendStrategy
from src.strategies.bb_squeeze import BBSqueezeStrategy
from src.strategies.ema_cross import EMACrossStrategy
from src.strategies.donchian import DonchianStrategy
from src.strategies.rsi_multiplier import RSIMultiplierStrategy
from src.strategies.rsi_binary import RSIBinaryStrategy
from src.strategies.bb_binary import BBBinaryStrategy
from src.strategies.mtf_v5 import MTFV5Strategy
from src.strategies.crash_boom_recoil import CrashBoomRecoilStrategy
from src.strategies.calm_accu import CalmAccuStrategy
from src.strategies.bb_multiplier import BBMultiplierStrategy
from src.strategies.digit_even import DigitEvenStrategy
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
    rotation="00:00",
    retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}",
)
logger.add(sys.stderr, level="ERROR", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Per-symbol context ─────────────────────────────────────────────────

@dataclass
class SymbolBot:
    symbol: str
    tick_store: TickStore
    strategy: BaseStrategy
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


def build_risk_config(symbol_entry: dict, base_risk: dict) -> dict:
    """Merge per-symbol risk overrides (max_stake, min_stake, etc.) onto global risk config."""
    cfg = dict(base_risk)
    risk_keys = {"max_stake", "min_stake", "stake_percent", "use_kelly",
                 "daily_loss_limit", "payout_pct", "max_open_contracts",
                 "blocked_hours_eat", "rolling_window", "min_rolling_wr"}
    for k in risk_keys:
        if k in symbol_entry:
            cfg[k] = symbol_entry[k]
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


# ── Tick watchdog ──────────────────────────────────────────────────────

async def tick_watchdog(bots: list[SymbolBot], stale_secs: int = 600):
    """
    Warn if any symbol's last tick is older than stale_secs (default 10 min).
    A stale symbol means the WebSocket stopped delivering data for that feed.
    """
    # seed with current epoch so we don't false-alarm on startup
    last_seen: dict[str, float] = {}
    for bot in bots:
        last_seen.setdefault(bot.symbol, _time.time())

    # patch each bot's tick handler to update last_seen
    originals: dict[str, list] = {}
    for sym in list(last_seen):
        originals[sym] = []

    # check every 2 minutes
    while True:
        await asyncio.sleep(120)
        for bot in bots:
            epoch = bot.tick_store.latest_epoch
            if epoch is not None:
                last_seen[bot.symbol] = max(last_seen.get(bot.symbol, 0), epoch)
            gap = _time.time() - last_seen.get(bot.symbol, _time.time())
            if gap > stale_secs:
                logger.warning(
                    f"[{bot.symbol}] No ticks for {gap/60:.1f} min — WebSocket may be stale"
                )


# ── Main bot loop ───────────────────────────────────────────────────────

async def run(watch_only: bool = False):
    config        = load_config()
    base_strategy = config["strategy"]
    risk_cfg      = config["risk"]

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

    num_symbols = len(active_entries)
    balance_per = total_balance / num_symbols
    logger.info(f"Trading {num_symbols} symbol(s) | {balance_per:.2f} balance allocated each")

    bots: list[SymbolBot] = []
    for entry in active_entries:
        sym_cfg  = build_symbol_config(entry, base_strategy)
        sym_risk = build_risk_config(entry, risk_cfg)
        symbol   = sym_cfg["symbol"]
        strategy_type = sym_cfg.get("strategy_type", "rsi_reversal")

        tick_store = TickStore(
            rsi_period=sym_cfg.get("rsi_period", 14),
            bar_size=sym_cfg.get("bar_size", 1),
        )

        # ── Strategy instantiation ────────────────────────────────
        granularity = sym_cfg.get("bar_seconds", 3600)

        if strategy_type == "gold_trend":
            strategy     = GoldTrendStrategy(sym_cfg)
            candle_count = sym_cfg.get("ema_period", 200) + sym_cfg.get("slope_bars", 5) + 20
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "macd_trend":
            strategy     = MACDTrendStrategy(sym_cfg)
            candle_count = sym_cfg.get("macd_slow", 26) + sym_cfg.get("macd_signal", 9) + 20
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "bb_squeeze":
            strategy     = BBSqueezeStrategy(sym_cfg)
            candle_count = sym_cfg.get("bb_period", 50) + 120
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "ema_cross":
            strategy     = EMACrossStrategy(sym_cfg)
            candle_count = sym_cfg.get("ema_slow", 50) + 20
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "donchian":
            strategy     = DonchianStrategy(sym_cfg)
            candle_count = sym_cfg.get("donchian_period", 30) + 10
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                from src.data.history import fetch_candles_async as _fetch_ohlcv
                candles = await _fetch_ohlcv(symbol, count=candle_count,
                                             granularity=granularity, return_full=True)
                strategy.seed_candles(candles)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(candles)} OHLCV bars")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "bb_multiplier":
            strategy     = BBMultiplierStrategy(sym_cfg)
            candle_count = sym_cfg.get("bb_period", 10) + 120
            logger.info(f"[{symbol}/bb_multiplier] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/bb_multiplier] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/bb_multiplier] Seed failed: {e} — warming up live")

        elif strategy_type == "rsi_multiplier":
            strategy     = RSIMultiplierStrategy(sym_cfg)
            candle_count = sym_cfg.get("rsi_period", 14) + 20
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "rsi_binary":
            strategy     = RSIBinaryStrategy(sym_cfg)
            candle_count = sym_cfg.get("rsi_period", 14) + 20
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "bb_binary":
            strategy     = BBBinaryStrategy(sym_cfg)
            candle_count = sym_cfg.get("bb_period", 10) + 120
            logger.info(f"[{symbol}/{strategy_type}] Seeding {candle_count} candles...")
            try:
                closes = await fetch_candles_async(symbol, count=candle_count, granularity=granularity)
                strategy.seed_candles(closes)
                logger.info(f"[{symbol}/{strategy_type}] Seeded {len(closes)} closes")
            except Exception as e:
                logger.warning(f"[{symbol}/{strategy_type}] Seed failed: {e} — warming up live")

        elif strategy_type == "crash_boom_recoil":
            # Crash/Boom spike-recoil: no candle seeding needed — warms up from live ticks.
            strategy = CrashBoomRecoilStrategy(sym_cfg)
            logger.info(f"[{symbol}/crash_boom_recoil] Ready — warms up from first "
                        f"{sym_cfg.get('atr_period', 50) + 2} live ticks")

        elif strategy_type == "calm_accu":
            # Inter-spike calm-period ACCU: no candle seeding needed — warms up from live ticks.
            strategy = CalmAccuStrategy(sym_cfg)
            logger.info(f"[{symbol}/calm_accu] Ready — warms up from first "
                        f"{sym_cfg.get('long_atr_period', 50) + 2} live ticks")

        elif strategy_type == "digit_even":
            # Structural even/odd digit bias in R_50/R_75 — no warm-up needed.
            strategy = DigitEvenStrategy(sym_cfg)
            logger.info(f"[{symbol}/digit_even] Ready — fires on every tick")

        elif strategy_type == "mtf_v5":
            strategy     = MTFV5Strategy(sym_cfg)
            htf_count    = sym_cfg.get("ema_period", 100) + sym_cfg.get("slope_bars", 3) + 20
            daily_count  = (sym_cfg.get("macro_ema_period", 20) + 10
                            if sym_cfg.get("macro_filter") else 0)
            logger.info(f"[{symbol}/mtf_v5] Seeding LTF=200 HTF={htf_count} Daily={daily_count}...")
            try:
                ltf_bars     = await fetch_candles_async(symbol, count=200,
                                                         granularity=300, return_full=True)
                htf_closes   = await fetch_candles_async(symbol, count=htf_count,
                                                         granularity=3600)
                daily_closes = (await fetch_candles_async(symbol, count=daily_count,
                                                          granularity=86400)
                                if daily_count else [])
                strategy.seed_all(ltf_bars, htf_closes, daily_closes)
                logger.info(f"[{symbol}/mtf_v5] Seeded LTF={len(ltf_bars)} "
                            f"HTF={len(htf_closes)} Daily={len(daily_closes)}")
            except Exception as e:
                logger.warning(f"[{symbol}/mtf_v5] Seed failed: {e} — warming up live")

        else:
            strategy = RSIReversalStrategy(sym_cfg)

        _is_multi = strategy_type in ("gold_trend", "macd_trend", "bb_squeeze",
                                       "ema_cross", "donchian", "rsi_multiplier",
                                       "bb_multiplier", "mtf_v5")

        risk = RiskManager(sym_risk, starting_balance=balance_per, symbol=symbol)

        trader = Trader(
            client, risk, symbol,
            sym_cfg.get("contract_duration", 10),
            sym_cfg.get("contract_duration_unit", "t"),
            multiplier=sym_cfg.get("multiplier", 0),
            growth_rate=sym_cfg.get("growth_rate", 0.03),
            hold_ticks=sym_cfg.get("hold_ticks", 5),
            early_sell_pct=sym_cfg.get("early_sell_pct", 0.0),
            strategy=strategy,
            alerter=alerter,
        )

        # Map any signal to CALL/PUT for paper-trade direction tracking
        _PAPER_CT = {"BUY_RISE": "CALL", "BUY_FALL": "PUT"}

        # Register per-symbol tick handler
        async def make_handler(b: SymbolBot, wonly: bool, contract_duration: int):
            async def handler(tick: dict):
                b.tick_store.add(tick)
                b.risk.check_auto_reset()
                current_price = float(tick["quote"])
                signal = b.strategy.evaluate(b.tick_store)
                if b.risk.is_halted:
                    ct  = _PAPER_CT.get(signal.action, "")
                    dur = signal.contract_duration if signal.contract_duration is not None \
                          else contract_duration
                    b.risk.on_tick_halted(current_price, signal.action, ct, dur)
                elif not wonly:
                    await b.trader.execute(signal)
            return handler

        bot = SymbolBot(symbol=symbol, tick_store=tick_store,
                        strategy=strategy, risk=risk, trader=trader)
        bots.append(bot)
        client.on_tick(symbol, await make_handler(
            bot, watch_only, sym_cfg.get("contract_duration", 10)
        ))

    # ── Dashboard (tracks first symbol for display) ───────────────
    primary = bots[0]
    dashboard = Dashboard(
        primary.tick_store,
        primary.risk,
        ema_period=base_strategy.get("ema_trend_period", 0),
    )

    mode_label    = "WATCH ONLY" if watch_only else "LIVE TRADING"
    symbols_label = ", ".join(b.symbol for b in bots)
    logger.info(f"Bot started | Mode: {mode_label} | Symbols: {symbols_label}")

    if alerter:
        await alerter.send_message(
            f"🤖 *Bot started*\nMode: {mode_label}\nSymbols: `{symbols_label}`\nBalance: `{total_balance:.2f} USD`"
        )

    dashboard.start()

    # ── Subscribe ticks for all symbols (once per unique symbol) ─
    seen_symbols: set[str] = set()
    for bot in bots:
        if bot.symbol not in seen_symbols:
            await client.subscribe_ticks(bot.symbol)
            seen_symbols.add(bot.symbol)

    # ── Start midnight reset loop + tick watchdog ─────────────────
    asyncio.create_task(midnight_reset_loop(bots, alerter))
    asyncio.create_task(tick_watchdog(bots))

    # ── Keep running ─────────────────────────────────────────────
    _last_halted_log = 0.0
    try:
        while True:
            dashboard.refresh()
            all_halted = all(b.risk.is_halted for b in bots)
            if all_halted:
                now = asyncio.get_event_loop().time()
                if now - _last_halted_log >= 300:
                    logger.info("All symbols halted — waiting for auto-reset.")
                    _last_halted_log = now
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
