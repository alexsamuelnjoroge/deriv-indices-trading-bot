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
    def __init__(self, rsi_period: int = 14, max_ticks: int = 500, bar_size: int = 1):
        self.rsi_period = rsi_period
        self.bar_size   = bar_size          # ticks per bar (1 = no aggregation)
        self._ticks: deque[Tick] = deque(maxlen=max_ticks)
        self._pending:    list[float]        = []                         # ticks in current bar
        self._bar_closes: deque[float]       = deque(maxlen=max_ticks)   # completed bar closes
        # RSI state (Wilder's smoothing — maintained incrementally)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._rsi: Optional[float] = None
        # Divergence: parallel history of (price, rsi) for every tick with valid RSI
        self._rsi_history: deque[tuple[float, float]] = deque(maxlen=max_ticks)

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

        if self.bar_size > 1:
            self._pending.append(tick.price)
            if len(self._pending) >= self.bar_size:
                self._bar_closes.append(self._pending[-1])
                self._pending.clear()
                self._update_rsi()
        else:
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
    def latest_epoch(self) -> Optional[int]:
        return self._ticks[-1].epoch if self._ticks else None

    @property
    def tick_count(self) -> int:
        return len(self._ticks)

    # ------------------------------------------------------------------ #
    #  RSI — Wilder's smoothed (incremental state)
    # ------------------------------------------------------------------ #

    def _update_rsi(self):
        prices = list(self._bar_closes) if self.bar_size > 1 else self.prices
        n = len(prices)

        if n < self.rsi_period + 1:
            self._rsi = None
            return

        if n == self.rsi_period + 1:
            changes = [prices[i] - prices[i - 1] for i in range(1, n)]
            self._avg_gain = sum(c for c in changes if c > 0) / self.rsi_period
            self._avg_loss = sum(abs(c) for c in changes if c < 0) / self.rsi_period
        else:
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

        if self._rsi is not None and self._ticks:
            self._rsi_history.append((self._ticks[-1].price, self._rsi))

    def rsi(self) -> Optional[float]:
        return self._rsi

    @property
    def warmup_ticks(self) -> int:
        """Raw ticks needed before RSI becomes available."""
        return self.bar_size * (self.rsi_period + 1)

    # ------------------------------------------------------------------ #
    #  Divergence detection
    # ------------------------------------------------------------------ #

    def bearish_divergence(self, lookback: int = 20) -> bool:
        """
        Bearish divergence: current price > a past price, but current RSI < past RSI.
        Signals weakening upward momentum — used to confirm BUY_FALL entries.
        """
        if len(self._rsi_history) < 2:
            return False
        cur_price, cur_rsi = self._rsi_history[-1]
        window = list(self._rsi_history)[max(0, len(self._rsi_history) - lookback - 1):-1]
        return any(cur_price > p and cur_rsi < r for p, r in window)

    def rsi_percentile(self, window: int, pct: float) -> Optional[float]:
        """
        Return the pct-th percentile of RSI values over the last `window` ticks.
        Used by adaptive threshold mode to set dynamic overbought/oversold levels.
        Requires at least 20 data points to be meaningful.
        """
        history = list(self._rsi_history)[-window:]
        if len(history) < 20:
            return None
        rsi_vals = sorted(r for _, r in history)
        idx = max(0, min(int(len(rsi_vals) * pct / 100), len(rsi_vals) - 1))
        return rsi_vals[idx]

    def bullish_divergence(self, lookback: int = 20) -> bool:
        """
        Bullish divergence: current price < a past price, but current RSI > past RSI.
        Signals weakening downward momentum — used to confirm BUY_RISE entries.
        """
        if len(self._rsi_history) < 2:
            return False
        cur_price, cur_rsi = self._rsi_history[-1]
        window = list(self._rsi_history)[max(0, len(self._rsi_history) - lookback - 1):-1]
        return any(cur_price < p and cur_rsi > r for p, r in window)

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

    def tick_streak(self) -> int:
        """
        Length of the current consecutive directional run.
        Returns +N for N consecutive up-ticks, -N for N consecutive down-ticks.
        A flat tick (same price) breaks the streak and returns 0.
        """
        prices = self.prices
        if len(prices) < 2:
            return 0
        if prices[-1] > prices[-2]:
            direction = 1
        elif prices[-1] < prices[-2]:
            direction = -1
        else:
            return 0
        streak = 1
        for i in range(len(prices) - 2, 0, -1):
            prev_dir = 1 if prices[i] > prices[i - 1] else (-1 if prices[i] < prices[i - 1] else 0)
            if prev_dir == direction:
                streak += 1
            else:
                break
        return streak * direction

    def zscore(self, period: int) -> Optional[float]:
        """
        Z-score of the current price relative to its rolling mean.
        Returns (price - MA_n) / std_n, or None during warmup.
        Positive = price above mean (overbought), negative = below (oversold).
        """
        prices = self.prices
        if len(prices) < period:
            return None
        window = prices[-period:]
        mean = sum(window) / period
        variance = sum((p - mean) ** 2 for p in window) / period
        std = variance ** 0.5
        if std == 0:
            return None
        return round((prices[-1] - mean) / std, 4)

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
