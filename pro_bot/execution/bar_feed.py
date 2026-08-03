"""
Bar Feed — polls MT5 for completed bars and fires callbacks on bar close.

Runs in an asyncio loop. Checks every `poll_interval` seconds whether a
new bar has closed. When a new bar closes, calls all registered handlers
with the full bar list (oldest first).
"""

import asyncio
import time
from typing import Callable

from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class BarFeed:

    def __init__(self, client, symbol: str, timeframe: int,
                 history_count: int = 500, poll_interval: float = 5.0):
        """
        client:         MT5Client instance
        symbol:         e.g. "AUDUSD"
        timeframe:      mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, etc.
        history_count:  bars to keep in memory
        poll_interval:  seconds between polls (default 5s for 5min bars)
        """
        self.client        = client
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.history_count = history_count
        self.poll_interval = poll_interval

        self._bars:     list[dict] = []
        self._last_bar_epoch: int  = 0
        self._handlers: list[Callable] = []
        self._running = False

    def on_bar_close(self, fn: Callable) -> None:
        """Register a callback: fn(symbol, bars) called on every new bar close."""
        self._handlers.append(fn)

    async def start(self) -> None:
        """Run the polling loop until stop() is called."""
        self._running = True
        logger.info(f"BarFeed started | {self.symbol} TF={self.timeframe} "
                    f"poll={self.poll_interval}s")

        # Initial load
        bars = self.client.get_bars(self.symbol, self.timeframe, self.history_count)
        if bars:
            self._bars           = bars
            self._last_bar_epoch = bars[-1]["epoch"]
            logger.info(f"[{self.symbol}] Loaded {len(bars)} bars | "
                        f"last: {self._last_bar_epoch}")

        while self._running:
            await asyncio.sleep(self.poll_interval)
            self._poll()

    def stop(self) -> None:
        self._running = False

    def _poll(self) -> None:
        bars = self.client.get_bars(self.symbol, self.timeframe, self.history_count)
        if not bars:
            return

        latest_epoch = bars[-2]["epoch"]   # -1 is the forming bar; -2 is last closed

        if latest_epoch > self._last_bar_epoch:
            self._last_bar_epoch = latest_epoch
            self._bars           = bars[:-1]   # exclude forming bar
            logger.debug(f"[{self.symbol}] New bar closed at epoch {latest_epoch}")
            for handler in self._handlers:
                try:
                    handler(self.symbol, self._bars)
                except Exception as e:
                    logger.error(f"Bar handler error [{self.symbol}]: {e}")

    @property
    def bars(self) -> list[dict]:
        return self._bars
