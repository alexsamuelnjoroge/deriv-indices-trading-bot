"""
Stores incoming price ticks and computes technical indicators.

Indicators available:
  rsi()            — Wilder's smoothed RSI (standard EMA-based method)
  ema(period)      — Exponential Moving Average for any period
  atr(period)      — Average True Range (using |close - prev_close| as TR for tick data)
  bollinger_bands()— Upper / Middle / Lower Bollinger Bands
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class Tick:
    epoch: int
    symbol: str
    price: float


class TickStore:
    def __init__(self, rsi_period: int = 14, max_ticks: int = 500):
        self.rsi_period = rsi_period
        self._ticks: deque[Tick] = deque(maxlen=max_ticks)
        # RSI state (Wilder's smoothing — maintained incrementally)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._rsi: Optional[float] = None

    # ------------------------------------------------------------------ #
    #  Tick ingestion
    # ------------------------------------------------------------------ #

    def add(self, raw: dict):
        tick = Tick(
            epoch=raw["epoch"],
            symbol=raw["symbol"],
            price=float(raw["quote"]),
        )
        self._ticks.append(tick)
        self._update_rsi()
        return tick

    # ------------------------------------------------------------------ #
    #  Basic accessors
    # ------------------------------------------------------------------ #

    @property
    def prices(self) -> list[float]:
        return [t.price for t in self._ticks]

    @property
    def latest_price(self) -> Optional[float]:
        return self._ticks[-1].price if self._ticks else None

    @property
    def tick_count(self) -> int:
        return len(self._ticks)

    # ------------------------------------------------------------------ #
    #  RSI — Wilder's smoothed (incremental state)
    # ------------------------------------------------------------------ #

    def _update_rsi(self):
        prices = self.prices
        n = len(prices)

        if n < self.rsi_period + 1:
            self._rsi = None
            return

        if n == self.rsi_period + 1:
            # Seed: simple average of the first rsi_period changes
            changes = [prices[i] - prices[i - 1] for i in range(1, n)]
            self._avg_gain = sum(c for c in changes if c > 0) / self.rsi_period
            self._avg_loss = sum(abs(c) for c in changes if c < 0) / self.rsi_period
        else:
            # Wilder's smoothing: α = 1 / rsi_period
            change = prices[-1] - prices[-2]
            gain = change if change > 0 else 0.0
            loss = abs(change) if change < 0 else 0.0
            self._avg_gain = (self._avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._avg_loss = (self._avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if self._avg_loss == 0:
            self._rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._rsi = round(100 - (100 / (1 + rs)), 2)

    def rsi(self) -> Optional[float]:
        return self._rsi

    # ------------------------------------------------------------------ #
    #  EMA — Exponential Moving Average
    # ------------------------------------------------------------------ #

    def ema(self, period: int) -> Optional[float]:
        """Standard EMA seeded with SMA of the first `period` prices."""
        prices = self.prices
        if len(prices) < period:
            return None
        k = 2.0 / (period + 1)
        val = sum(prices[:period]) / period
        for price in prices[period:]:
            val = price * k + val * (1 - k)
        return round(val, 5)

    # ------------------------------------------------------------------ #
    #  ATR — Average True Range (tick data: TR = |close - prev_close|)
    # ------------------------------------------------------------------ #

    def atr(self, period: int = 14) -> Optional[float]:
        """
        ATR using Wilder's smoothing on absolute tick-to-tick changes.
        Tick data has no high/low, so |close[i] - close[i-1]| is the True Range.
        """
        prices = self.prices
        if len(prices) < period + 1:
            return None
        ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        avg = sum(ranges[:period]) / period
        for r in ranges[period:]:
            avg = (avg * (period - 1) + r) / period
        return round(avg, 7)

    # ------------------------------------------------------------------ #
    #  Bollinger Bands
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Simple RSI — for any period, computed on-demand (for dual-RSI)
    # ------------------------------------------------------------------ #

    def rsi_simple(self, period: int) -> Optional[float]:
        """
        RSI using a plain average (not Wilder's smoothing).
        Slightly less stable than the incremental rsi() but works for any
        period without pre-warming — used as a secondary confirmation RSI.
        """
        prices = self.prices
        if len(prices) < period + 1:
            return None
        window  = prices[-(period + 1):]
        changes = [window[i] - window[i - 1] for i in range(1, len(window))]
        avg_gain = sum(c for c in changes if c > 0) / period
        avg_loss = sum(abs(c) for c in changes if c < 0) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    # ------------------------------------------------------------------ #
    #  Bollinger Bands
    # ------------------------------------------------------------------ #

    def bollinger_bands(
        self, period: int = 20, num_std: float = 2.0
    ) -> Optional[tuple[float, float, float]]:
        """Returns (upper, middle, lower) bands using the last `period` prices."""
        prices = self.prices
        if len(prices) < period:
            return None
        window = prices[-period:]
        middle = sum(window) / period
        variance = sum((p - middle) ** 2 for p in window) / period
        std = variance ** 0.5
        return (
            round(middle + num_std * std, 5),
            round(middle, 5),
            round(middle - num_std * std, 5),
        )
