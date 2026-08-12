"""
pro_bot/main.py — MT4/MT5 strategy runner.

Wires together:
  BarFeed (bar-close events) → strategies → OrderManager → MT5Client

Usage:
    python -m pro_bot.main [--config pro_bot/config.yaml]

Setup:
    1. Install MT5 terminal + set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env
    2. Enable "Allow automated trading" in MT5 Expert Advisors settings
    3. pip install MetaTrader5 loguru python-dotenv pyyaml
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from pro_bot.execution.mt5_client import MT5Client
from pro_bot.execution.bar_feed   import BarFeed
from pro_bot.execution.order_manager import OrderManager
from pro_bot.strategies.mtf_pullback     import MTFPullbackStrategy
from pro_bot.strategies.bb_rsi_pullback  import BBRSIPullbackStrategy
from pro_bot.strategies.macd_momentum    import MACDMomentumStrategy
from pro_bot.strategies.stoch_ema        import StochEMAStrategy
from pro_bot.strategies.session_breakout import SessionBreakoutStrategy
from pro_bot.strategies.rsi_divergence   import RSIDivergenceStrategy
from pro_bot.strategies.pivot_bounce     import PivotBounceStrategy

try:
    import MetaTrader5 as mt5
    MT5_TF = {
        300:   mt5.TIMEFRAME_M5,
        900:   mt5.TIMEFRAME_M15,
        3600:  mt5.TIMEFRAME_H1,
        14400: mt5.TIMEFRAME_H4,
        86400: mt5.TIMEFRAME_D1,
    }
    GRAN_4H = 14400
except ImportError:
    mt5    = None
    MT5_TF = {}
    GRAN_4H = 14400

STRATEGY_CLASSES = {
    "mtf_pullback":      MTFPullbackStrategy,
    "bb_rsi_pullback":   BBRSIPullbackStrategy,
    "macd_momentum":     MACDMomentumStrategy,
    "stoch_ema":         StochEMAStrategy,
    "session_breakout":  SessionBreakoutStrategy,
    "rsi_divergence":    RSIDivergenceStrategy,
    "pivot_bounce":      PivotBounceStrategy,
}

# Strategies that need LTF + HTF + optional daily feeds (same wiring as mtf_pullback)
_MTF_STRATEGIES = {"mtf_pullback", "bb_rsi_pullback", "macd_momentum"}


def _setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add("logs/pro_bot_{time:YYYYMMDD}.log", level="DEBUG", rotation="1 day",
               retention="14 days")
    Path("logs").mkdir(exist_ok=True)


class ProBot:

    def __init__(self, config: dict):
        self.config  = config
        self.risk_cfg = config.get("risk", {})
        self.client  = MT5Client()
        self.order_mgr = OrderManager(self.client, self.risk_cfg)
        self._feeds: list[BarFeed] = []

    # ── Build feeds + strategy handlers ─────────────────────────────────────

    def _build(self):
        for strat_cfg in self.config.get("strategies", []):
            if not strat_cfg.get("enabled", True):
                continue

            name     = strat_cfg["name"]
            symbols  = strat_cfg.get("symbols", [])
            cls      = STRATEGY_CLASSES.get(name)
            if cls is None:
                logger.warning(f"Unknown strategy: {name}")
                continue

            granularity = strat_cfg.get("granularity",     900)
            ltf_gran    = strat_cfg.get("ltf_granularity", granularity)
            htf_gran    = strat_cfg.get("htf_granularity", 3600)

            ltf_tf = MT5_TF.get(ltf_gran)
            htf_tf = MT5_TF.get(htf_gran)
            if ltf_tf is None:
                logger.warning(f"[{name}] Unsupported granularity {ltf_gran}s")
                continue

            for symbol in symbols:
                strategy = cls(strat_cfg)

                # --- MTF-style strategies need LTF + HTF feeds + optional daily ---
                if name in _MTF_STRATEGIES:
                    ltf_feed = BarFeed(self.client, symbol, ltf_tf)
                    htf_feed = BarFeed(self.client, symbol, htf_tf,
                                       history_count=300)

                    daily_feed = None
                    if strat_cfg.get("macro_filter"):
                        daily_tf = MT5_TF.get(86400)
                        if daily_tf:
                            daily_feed = BarFeed(self.client, symbol, daily_tf,
                                                 history_count=100)

                    _4h_feed = None
                    if strat_cfg.get("use_4h_filter"):
                        tf_4h = MT5_TF.get(GRAN_4H)
                        if tf_4h:
                            _4h_feed = BarFeed(self.client, symbol, tf_4h,
                                               history_count=200)

                    def _make_mtf_handler(sym, strat, h_feed, d_feed, feed_4h):
                        _ltf_seeded   = [False]
                        _htf_seeded   = [False]
                        _daily_seeded = [False]
                        _4h_seeded    = [False]

                        def _ltf_handler(symbol_name, bars):
                            if not bars:
                                return
                            if not _ltf_seeded[0]:
                                strat._bars = list(bars[:-1])
                                _ltf_seeded[0] = True
                            sig = strat.feed(bars[-1])
                            if sig and sig.action != "HOLD":
                                self.order_mgr.on_signal(sym, sig)

                        def _htf_handler(symbol_name, bars):
                            if not bars:
                                return
                            if not _htf_seeded[0]:
                                for bar in bars[:-1]:
                                    strat.feed_htf(bar)
                                _htf_seeded[0] = True
                            strat.feed_htf(bars[-1])

                        h_feed.on_bar_close(_htf_handler)

                        if feed_4h:
                            def _4h_handler(symbol_name, bars):
                                if not bars:
                                    return
                                if not _4h_seeded[0]:
                                    for bar in bars[:-1]:
                                        strat.feed_4h(bar)
                                    _4h_seeded[0] = True
                                strat.feed_4h(bars[-1])
                            feed_4h.on_bar_close(_4h_handler)

                        if d_feed:
                            def _daily_handler(symbol_name, bars):
                                if not bars:
                                    return
                                if not _daily_seeded[0]:
                                    for bar in bars[:-1]:
                                        strat.feed_daily(bar)
                                    _daily_seeded[0] = True
                                strat.feed_daily(bars[-1])
                            d_feed.on_bar_close(_daily_handler)

                        return _ltf_handler

                    ltf_feed.on_bar_close(
                        _make_mtf_handler(symbol, strategy, htf_feed, daily_feed, _4h_feed)
                    )
                    self._feeds.append(ltf_feed)
                    self._feeds.append(htf_feed)
                    if _4h_feed:
                        self._feeds.append(_4h_feed)
                    if daily_feed:
                        self._feeds.append(daily_feed)

                # --- Pivot Bounce needs daily bars too ---
                elif name == "pivot_bounce":
                    ltf_feed  = BarFeed(self.client, symbol, ltf_tf)
                    daily_tf  = MT5_TF.get(86400)
                    daily_feed = BarFeed(self.client, symbol, daily_tf,
                                        history_count=60) if daily_tf else None

                    def _make_pivot_handler(sym, strat, d_feed):
                        if d_feed:
                            def _daily_handler(symbol_name, bars):
                                if bars:
                                    strat.feed_daily_bar(bars[-1])
                            d_feed.on_bar_close(_daily_handler)

                        def _handler(symbol_name, bars):
                            sig = strat.feed(bars[-1] if bars else {})
                            if sig and sig.action != "HOLD":
                                self.order_mgr.on_signal(sym, sig)

                        return _handler

                    ltf_feed.on_bar_close(_make_pivot_handler(symbol, strategy, daily_feed))
                    self._feeds.append(ltf_feed)
                    if daily_feed:
                        self._feeds.append(daily_feed)

                # --- All other strategies: single timeframe ---
                else:
                    tf   = MT5_TF.get(granularity, ltf_tf)
                    feed = BarFeed(self.client, symbol, tf)

                    def _make_handler(sym, strat):
                        def _handler(symbol_name, bars):
                            sig = strat.feed(bars[-1] if bars else {})
                            if sig and sig.action != "HOLD":
                                self.order_mgr.on_signal(sym, sig)
                        return _handler

                    feed.on_bar_close(_make_handler(symbol, strategy))
                    self._feeds.append(feed)

    # ── Monitoring loop ──────────────────────────────────────────────────────

    async def _monitor(self):
        """Every 30 s: sync positions, log status, check for halted state."""
        while True:
            await asyncio.sleep(30)
            try:
                self.order_mgr.sync_positions()
                self.order_mgr.manage_positions()
                s = self.order_mgr.status()
                balance = self.client.get_balance()
                equity  = self.client.get_equity()
                open_str = (f"{s['open_trades']}+{s['open_pyramids']}pyr"
                            if s['open_pyramids'] else str(s['open_trades']))
                logger.info(
                    f"STATUS | Balance ${balance:.2f} Equity ${equity:.2f} | "
                    f"Open {open_str} | Today P&L ${s['today_pnl']:+.2f} | "
                    f"W/L {s['wins']}/{s['losses']} WR {s['win_rate']}% | "
                    f"{'HALTED' if s['halted'] else 'ACTIVE'}"
                )
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    # ── Main entry ───────────────────────────────────────────────────────────

    async def run(self):
        if not self.client.connect():
            logger.error("Cannot connect to MT5. Is terminal running?")
            return

        self._build()

        if not self._feeds:
            logger.error("No feeds built — check config.yaml and enabled strategies")
            return

        logger.info(f"Starting {len(self._feeds)} feed(s)")
        tasks = [asyncio.create_task(f.start()) for f in self._feeds]
        tasks.append(asyncio.create_task(self._monitor()))

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            for f in self._feeds:
                f.stop()
            self.client.disconnect()
            s = self.order_mgr.status()
            logger.info(f"Session ended | Total trades: {s['total_trades']} | "
                        f"W/L {s['wins']}/{s['losses']} | P&L ${s['today_pnl']:+.2f}")


def main():
    load_dotenv()
    _setup_logging()

    parser = argparse.ArgumentParser(description="pro_bot MT5 runner")
    parser.add_argument("--config", default="pro_bot/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if mt5 is None:
        logger.error(
            "MetaTrader5 package not found.\n"
            "Install with: pip install MetaTrader5\n"
            "Note: Windows only — requires MT5 terminal installed."
        )
        sys.exit(1)

    bot = ProBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
