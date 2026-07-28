"""
Donchian Channel Breakout Strategy.

Validated on frxXAUUSD 1h bars (walk-forward 3/3 folds):
  Don(30): bar high > 30-bar channel high -> MULTUP
           bar low  < 30-bar channel low  -> MULTDOWN
  SL=1.00% TP=2.00% | BE WR=34.0%
  Walk-forward mean EV = +0.308

Tracks bar high/low live from the tick stream so the channel matches
the OHLCV Donchian used during backtesting.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class DonchianStrategy(BaseStrategy):

    def __init__(self, config: dict):
        super().__init__(config)
        self.channel_period = config.get("donchian_period", 30)
        self.sl_pct         = config.get("sl_pct",         0.010)
        self.tp_pct         = config.get("tp_pct",         0.020)
        self.bar_seconds    = config.get("bar_seconds",     3600)

        self._bar_highs:  list[float]   = []
        self._bar_lows:   list[float]   = []
        self._bar_closes: list[float]   = []
        self._bar_start:  Optional[int] = None
        self._bar_high:   Optional[float] = None
        self._bar_low:    Optional[float] = None

    def seed_candles(self, candles: list[dict]) -> None:
        """Pre-load historical candles (list of {high, low, close}) oldest-first."""
        self._bar_highs  = [float(c["high"])  for c in candles]
        self._bar_lows   = [float(c["low"])   for c in candles]
        self._bar_closes = [float(c["close"]) for c in candles]

    # ------------------------------------------------------------------ #

    def evaluate(self, tick_store) -> Signal:
        price = tick_store.latest_price
        epoch = tick_store.latest_epoch
        if price is None or epoch is None:
            return Signal(action="HOLD", reason="No ticks yet")

        # Track intra-bar high/low from tick stream
        if self._bar_high is None or price > self._bar_high:
            self._bar_high = price
        if self._bar_low is None or price < self._bar_low:
            self._bar_low = price

        if self._bar_start is None:
            self._bar_start = epoch
            return Signal(action="HOLD", reason="Bar started")

        elapsed = epoch - self._bar_start
        if elapsed < self.bar_seconds:
            return Signal(action="HOLD", reason=f"In bar ({elapsed}s/{self.bar_seconds}s)")

        # Bar close — record OHLCV and reset intra-bar trackers
        self._bar_highs.append(self._bar_high)
        self._bar_lows.append(self._bar_low)
        self._bar_closes.append(price)
        self._bar_high  = None
        self._bar_low   = None
        self._bar_start = epoch

        if len(self._bar_highs) <= self.channel_period:
            needed = self.channel_period + 1
            return Signal(action="HOLD",
                          reason=f"Warming ({len(self._bar_highs)}/{needed})")

        return self._evaluate_bar()

    def _evaluate_bar(self) -> Signal:
        n  = self.channel_period
        # Channel is the highest high / lowest low of the PREVIOUS n bars (excluding current)
        ch_high = max(self._bar_highs[-(n + 1):-1])
        ch_low  = min(self._bar_lows [-(n + 1):-1])
        bar_h   = self._bar_highs[-1]
        bar_l   = self._bar_lows[-1]

        if bar_h > ch_high:
            return Signal(
                action="BUY_RISE",
                reason=f"Donchian breakout up (bar_h={bar_h:.4f} > ch={ch_high:.4f})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )
        if bar_l < ch_low:
            return Signal(
                action="BUY_FALL",
                reason=f"Donchian breakout dn (bar_l={bar_l:.4f} < ch={ch_low:.4f})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
            )

        return Signal(action="HOLD",
                      reason=f"No breakout (ch_h={ch_high:.4f} ch_l={ch_low:.4f})")

    def on_result(self, won: bool) -> None:
        pass
